import os
import json
import time
import asyncio
import hashlib
import uuid
from typing import AsyncGenerator, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()


# ==================== 🛠️ 工具类：图谱管理器 ====================
class GraphManager:
    """负责与 Neo4j 交互的所有底层操作"""

    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _generate_id(self, name: str) -> str:
        """生成确定性ID"""
        return f"node_{hashlib.md5(name.encode('utf-8')).hexdigest()[:8]}"

    def _assign_color(self, group: str) -> str:
        """
        根据实体类型分配颜色 (支持中文映射和模糊匹配)
        增强适配性：例如 '人物'、'开发者'、'创始人' 都会被映射为 Person 的颜色
        """
        # 1. 定义基础色盘 (方便统一修改)
        colors = {
            "orange": "#ff9900",   # 概念/核心
            "pink": "#ff66cc",     # 人物
            "blue": "#66ccff",     # 作品/电影
            "cyan": "#00cc99",     # 技术
            "purple": "#9966ff",   # 地点
            "dark_blue": "#4d4dff",# 公司/组织
            "grey": "#808080",     # 时间/其他
            "default": "#cccccc"   # 未知
        }

        # 2. 定义关键词映射规则 (Key = 关键词, Value = 色盘Key)
        # 你可以在这里随意添加 DeepSeek 可能输出的中文词汇
        mapping_rules = {
            # === 人物类 ===
            "Person": "pink", "人": "pink", "人物": "pink", "用户": "pink",
            "演员": "pink", "导演": "pink", "开发者": "pink", "创始人": "pink", 
            "CEO": "pink", "专家": "pink", "作者": "pink",
            
            # === 影视/作品类 ===
            "Movie": "blue", "电影": "blue", "影片": "blue", "作品": "blue", 
            "书籍": "blue", "小说": "blue", "电视剧": "blue",
            
            # === 概念/技术类 ===
            "Concept": "orange", "概念": "orange", "术语": "orange", "定义": "orange",
            "Technology": "cyan", "技术": "cyan", "科技": "cyan", "工具": "cyan", "语言": "cyan",
            
            # === 地点类 ===
            "Location": "purple", "地点": "purple", "城市": "purple", "国家": "purple", "地址": "purple",
            
            # === 公司/组织类 ===
            "Company": "dark_blue", "公司": "dark_blue", "企业": "dark_blue", 
            "机构": "dark_blue", "品牌": "dark_blue", "集团": "dark_blue",
            
            # === 时间类 ===
            "Date": "grey", "时间": "grey", "日期": "grey", "年份": "grey", "年代": "grey"
        }
        
        # 3. 匹配逻辑
        if not group:
            return colors["default"]
            
        # 3.1 尝试直接精确匹配 (最快)
        # 例如: group="开发者" -> 命中 -> 返回 pink
        if group in mapping_rules:
            return colors[mapping_rules[group]]
            
        # 3.2 尝试模糊匹配 (包含关系)
        # 例如: group="科幻电影" -> 包含 "电影" -> 返回 blue
        # 例如: group="著名人物" -> 包含 "人物" -> 返回 pink
        for key, color_key in mapping_rules.items():
            if key in group:
                return colors[color_key]
                
        # 4. 如果都匹配不上，返回默认灰色
        return colors["default"]

    def search_subgraph(self, keywords: List[str]) -> str:
        """
        【RAG 核心优化版】：精确查询实体间的具体关系
        """
        if not keywords:
            return ""

        context_texts = []
        with self.driver.session() as session:
            # === 修改点 1: 查询具体的 name 属性 ===
            # 我们之前查询的是 type(r)，那只是 "RELATED"。
            # 现在我们查询 r.name，这里存储了 "导演"、"作者" 等具体含义。
            query = """
            UNWIND $keywords AS kw
            MATCH (n:Entity) WHERE n.name CONTAINS kw
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name, r.name, m.name
            LIMIT 30
            """
            result = session.run(query, keywords=keywords)

            seen = set()
            for record in result:
                n_name = record["n.name"]
                rel_name = record["r.name"]  # 取出具体的“作者”、“导演”
                m_name = record["m.name"]

                if rel_name and m_name:
                    # 拼接成自然语言，例如：流浪地球 --[导演]--> 郭帆
                    fact = f"{n_name} 的关系是 [{rel_name}] 对象是 {m_name}"
                else:
                    fact = f"实体: {n_name}"

                if fact not in seen:
                    context_texts.append(fact)
                    seen.add(fact)

        return "\n".join(context_texts)

    def update_graph(self, session_id: str, extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """将提取的 JSON 数据写入 Neo4j"""
        raw_nodes = extraction_data.get("entities", [])
        raw_relations = extraction_data.get("relations", [])

        frontend_update_data = {"nodes": [], "links": []}

        if not raw_nodes and not raw_relations:
            return frontend_update_data

        with self.driver.session() as session:
            # 1. 节点写入 (保持不变)
            for item in raw_nodes:
                if not isinstance(item, dict) or 'name' not in item: continue

                node_id = self._generate_id(item['name'])
                group = item.get('type', 'Concept')

                frontend_node = {
                    "id": node_id, "name": item['name'], "group": group,
                    "color": self._assign_color(group), "val": 15
                }
                frontend_update_data["nodes"].append(frontend_node)

                session.run("""
                    MERGE (n:Entity {id: $id})
                    SET n.name = $name, n.group = $group, n.color = $color
                """, **frontend_node)

            # 2. 关系写入 (保持不变，但强调 r.name 的重要性)
            for item in raw_relations:
                if not isinstance(item, dict) or 'head' not in item or 'tail' not in item: continue

                source_id = self._generate_id(item['head'])
                target_id = self._generate_id(item['tail'])
                rel_name = item.get('relation', '关联')  # 具体的“作者”、“位于”

                frontend_link = {
                    "source": source_id, "target": target_id,
                    "relationship": rel_name, "width": 2
                }
                frontend_update_data["links"].append(frontend_link)

                # === 关键逻辑 ===
                # 我们依然使用通用的 :RELATED 类型，因为 Neo4j 无法参数化关系类型。
                # 但是我们将具体的 rel_name 存入 {name: $rel} 属性中。
                session.run("""
                    MATCH (s:Entity {id: $sid}), (t:Entity {id: $tid})
                    MERGE (s)-[r:RELATED {name: $rel}]->(t)
                """, sid=source_id, tid=target_id, rel=rel_name)

        return frontend_update_data


# ==================== 🧠 核心类：双脑处理器 ====================
class DualBrain:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.graph_manager = GraphManager()

    async def _extract_search_keywords(self, user_prompt: str) -> List[str]:
        """【后台脑】：提取关键词"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "提取用户输入中的核心实体名称。只输出关键词，用逗号分隔。"},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False
            )
            content = response.choices[0].message.content
            return [k.strip() for k in content.replace("，", ",").split(",") if k.strip()]
        except Exception:
            return []

    async def _fast_brain_generate(self, user_prompt: str, context: str) -> AsyncGenerator[str, None]:
        """【前台脑】：基于具体关系上下文回答"""

        # === 修改点 2: 增强 System Prompt ===
        system_prompt = "你是一个基于知识图谱的智能助手。"
        if context:
            system_prompt += f"\n\n【知识图谱检索结果】:\n{context}\n\n请注意：上文中的'关系是 [XXX]'表示实体间的具体联系。请依据这些具体关系回答用户。"
        else:
            system_prompt += "\n\n当前知识库中没有相关信息，请利用你的通用知识回答。"

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.7
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'type': 'chunk', 'id': f'resp_{uuid.uuid4().hex[:6]}', 'content': chunk.choices[0].delta.content})}\n\n"
                await asyncio.sleep(0.01)

    async def _slow_brain_learn(self, session_id: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """【后台脑】：提取具体关系"""
        yield f"data: {json.dumps({'type': 'control', 'status': 'thinking', 'payload': '正在分析具体关系...'})}\n\n"

        # === 修改点 3: 优化提取 Prompt ===
        # 强制要求提取具体的动词或名词关系，而不是笼统的“相关”
        extraction_prompt = f"""
        请分析用户的输入，提取事实性三元组。
        用户输入："{user_prompt}"

        要求：
        1. 如果是提问，返回空列表。
        2. 如果是陈述事实，提取实体和**具体关系**。
        3. **关系(relation)** 必须是具体的动词或名词，例如："导演"、"作者"、"位于"、"属于"、"CEO"。不要使用 "相关"、"联系" 这种模糊词。
        4. 返回纯 JSON:
        {{
            "entities": [{{"name": "实体名", "type": "类型"}}],
            "relations": [{{"head": "头实体", "tail": "尾实体", "relation": "具体关系词"}}]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            extracted_json = json.loads(content)

            # 容错
            if "entities" not in extracted_json: extracted_json["entities"] = []
            if "relations" not in extracted_json: extracted_json["relations"] = []

            # 写入数据库
            loop = asyncio.get_running_loop()
            frontend_data = await loop.run_in_executor(
                None,
                self.graph_manager.update_graph,
                session_id,
                extracted_json
            )

            if frontend_data["nodes"] or frontend_data["links"]:
                graph_event = {
                    "type": "graph_update", "action": "merge",
                    "data": frontend_data, "timestamp": int(time.time() * 1000)
                }
                yield f"data: {json.dumps(graph_event)}\n\n"
                yield f"data: {json.dumps({'type': 'control', 'status': 'finish', 'stop_reason': 'learned', 'summary': {'newNodes': len(frontend_data['nodes'])}})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'control', 'status': 'finish', 'payload': '未发现新知识'})}\n\n"

        except Exception as e:
            print(f"学习过程出错: {e}")
            yield f"data: {json.dumps({'type': 'control', 'status': 'error', 'payload': str(e)})}\n\n"

    async def think(self, session_id: str, user_prompt: str) -> AsyncGenerator[str, None]:
        
        yield f"data: {json.dumps({'type': 'control', 'status': 'start'})}\n\n"

        yield f"data: {json.dumps({'type': 'control', 'status': 'thinking', 'payload': '正在检索具体关系...'})}\n\n"
        keywords = await self._extract_search_keywords(user_prompt)
        loop = asyncio.get_running_loop()
        graph_context = await loop.run_in_executor(None, self.graph_manager.search_subgraph, keywords)

        if graph_context:
            yield f"data: {json.dumps({'type': 'control', 'status': 'thinking', 'payload': '已加载关联知识'})}\n\n"

        async for event in self._fast_brain_generate(user_prompt, graph_context):
            yield event

        async for event in self._slow_brain_learn(session_id, user_prompt):
            yield event

        yield f"data: {json.dumps({'type': 'control', 'status': 'closed'})}\n\n"

    def close(self):
        self.graph_manager.close()