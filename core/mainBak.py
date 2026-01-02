
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import time
import uuid

# ==================== 数据模型 ====================
class ChatMessage(BaseModel):
    sessionId: str
    content: str
    timestamp: Optional[int] = None

class GraphNode(BaseModel):
    id: str
    name: str
    group: str
    color: Optional[str] = None
    val: Optional[int] = None
    info: Optional[str] = None

class GraphLink(BaseModel):
    source: str
    target: str
    name: Optional[str] = None
    relationship: Optional[str] = None
    width: Optional[int] = None

class GraphUpdate(BaseModel):
    action: str  # merge | remove
    data: Dict[str, Any]

# ==================== 应用初始化 ====================
app = FastAPI(
    title="EvoMind API",
    version="1.0",
    description="动态知识图谱对话系统"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 模拟数据库 ====================
class MockDatabase:
    def __init__(self):
        self.sessions = {}
        self.graphs = {
            "default": {
                "nodes": [
                    {"id": "n1", "name": "知识图谱", "group": "Concept", "color": "#ff0000", "val": 20},
                    {"id": "n2", "name": "实体", "group": "Element", "color": "#00ff00", "val": 15},
                    {"id": "n3", "name": "关系", "group": "Element", "color": "#0000ff", "val": 15}
                ],
                "links": [
                    {"source": "n1", "target": "n2", "relationship": "包含", "width": 2},
                    {"source": "n1", "target": "n3", "relationship": "包含", "width": 2}
                ]
            }
        }
    
    def get_graph(self, session_id: str = "default"):
        """获取图谱数据"""
        return self.graphs.get(session_id, self.graphs["default"])
    
    def update_graph(self, session_id: str, update_data: Dict):
        """更新图谱（模拟）"""
        if session_id not in self.graphs:
            self.graphs[session_id] = {"nodes": [], "links": []}
        
        graph = self.graphs[session_id]
        
        # 合并新节点
        if "nodes" in update_data:
            for new_node in update_data["nodes"]:
                # 检查是否已存在
                existing = next((n for n in graph["nodes"] if n["id"] == new_node["id"]), None)
                if not existing:
                    graph["nodes"].append(new_node)
        
        # 合并新关系
        if "links" in update_data:
            for new_link in update_data["links"]:
                # 简单去重
                link_key = f"{new_link['source']}-{new_link['target']}-{new_link.get('relationship', '')}"
                existing = next(
                    (l for l in graph["links"] 
                     if f"{l['source']}-{l['target']}-{l.get('relationship', '')}" == link_key),
                    None
                )
                if not existing:
                    graph["links"].append(new_link)
        
        return graph

db = MockDatabase()

# ==================== 模拟AI处理器 ====================
class MockAIProcessor:
    """模拟AI处理消息并生成图谱更新"""
    
    @staticmethod
    def process_message(message: str) -> Dict[str, Any]:
        """根据消息内容生成模拟的图谱更新"""
        message_lower = message.lower()
        
        if "知识图谱" in message:
            return {
                "nodes": [
                    {"id": f"kg_{uuid.uuid4().hex[:4]}", "name": "语义网络", "group": "Concept", "color": "#ff9900", "val": 15}
                ],
                "links": [
                    {"source": "n1", "target": f"kg_{uuid.uuid4().hex[:4]}", "relationship": "属于", "width": 1}
                ]
            }
        elif "实体" in message:
            return {
                "nodes": [
                    {"id": f"ent_{uuid.uuid4().hex[:4]}", "name": "新实体", "group": "Element", "color": "#00cc99", "val": 10}
                ],
                "links": [
                    {"source": "n2", "target": f"ent_{uuid.uuid4().hex[:4]}", "relationship": "示例", "width": 1}
                ]
            }
        elif "关系" in message:
            return {
                "nodes": [
                    {"id": f"rel_{uuid.uuid4().hex[:4]}", "name": "新关系", "group": "Element", "color": "#9966ff", "val": 10}
                ],
                "links": [
                    {"source": "n3", "target": f"rel_{uuid.uuid4().hex[:4]}", "relationship": "类型", "width": 1}
                ]
            }
        else:
            # 默认添加一个随机节点
            node_id = f"node_{uuid.uuid4().hex[:4]}"
            return {
                "nodes": [
                    {"id": node_id, "name": f"概念{node_id[-2:]}", "group": "Concept", "color": "#cccccc", "val": 8}
                ],
                "links": [
                    {"source": "n1", "target": node_id, "relationship": "关联", "width": 1}
                ]
            }

# ==================== API接口 ====================

@app.get("/")
async def root():
    """服务状态检查"""
    return {
        "service": "EvoMind API",
        "status": "running",
        "endpoints": {
            "chat": "POST /api/chat",
            "stream": "GET /api/stream?sessionId=xxx",
            "snapshot": "GET /api/graph/snapshot?sessionId=xxx"
        },
        "note": "新手学习项目 - 动态知识图谱系统"
    }

@app.post("/api/chat")
async def process_chat(message: ChatMessage):
    """
    接收用户消息
    实际项目：这里会调用真正的AI模型
    """
    print(f"[API] 收到消息: {message.sessionId} - {message.content[:50]}...")
    
    # 模拟处理延迟
    await asyncio.sleep(0.5)
    
    return {
        "code": 200,
        "message": "消息已接收，开始处理",
        "data": {
            "sessionId": message.sessionId,
            "contentLength": len(message.content),
            "processing": True,
            "streamUrl": f"/api/stream?sessionId={message.sessionId}"
        }
    }

@app.get("/api/graph/snapshot")
async def get_graph_snapshot(sessionId: Optional[str] = None):
    """
    获取当前图谱快照
    用于页面初始化
    """
    graph_data = db.get_graph(sessionId or "default")
    
    return JSONResponse(content={
        "code": 200,
        "data": graph_data,
        "message": f"获取会话 {sessionId or 'default'} 的图谱成功",
        "stats": {
            "nodes": len(graph_data["nodes"]),
            "links": len(graph_data["links"])
        }
    })

@app.get("/api/stream")
async def stream_updates(sessionId: str):
    """
    SSE流式接口 - 核心功能
    推送AI响应和图谱实时更新
    """
    async def generate_events():
        """生成SSE事件流"""
        
        # 1. 发送开始信号
        start_event = {
            "type": "control",
            "status": "start",
            "payload": "开始处理您的请求"
        }
        yield f"data: {json.dumps(start_event)}\n\n"
        await asyncio.sleep(0.5)
        
        # 2. 发送思考状态
        thinking_event = {
            "type": "control",
            "status": "thinking",
            "payload": "正在分析语义关系..."
        }
        yield f"data: {json.dumps(thinking_event)}\n\n"
        await asyncio.sleep(1)
        
        # 3. 模拟AI响应文本（分块）
        ai_response = "知识图谱是一种用图模型来描述知识和建模事物之间关联关系的技术。它由节点（实体）和边（关系）组成，能够表示丰富的语义信息。"
        
        for i, char in enumerate(ai_response):
            chunk_event = {
                "type": "chunk",
                "id": f"resp_{sessionId[:6]}_{i}",
                "content": char,
                "index": i,
                "total": len(ai_response)
            }
            yield f"data: {json.dumps(chunk_event)}\n\n"
            await asyncio.sleep(0.03)  # 打字机效果
        
        # 4. 模拟图谱更新（分多次）
        update_steps = [
            {
                "action": "merge",
                "data": {
                    "nodes": [
                        {"id": "n4", "name": "属性", "group": "Element", "color": "#ff66cc", "val": 12}
                    ],
                    "links": [
                        {"source": "n2", "target": "n4", "relationship": "具有", "width": 1.5}
                    ]
                }
            },
            {
                "action": "merge",
                "data": {
                    "nodes": [
                        {"id": "n5", "name": "类型", "group": "Meta", "color": "#66ccff", "val": 10}
                    ],
                    "links": [
                        {"source": "n4", "target": "n5", "relationship": "属于", "width": 1}
                    ]
                }
            }
        ]
        
        for step in update_steps:
            # 先更新数据库（模拟）
            db.update_graph(sessionId, step["data"])
            
            # 发送图谱更新事件
            graph_event = {
                "type": "graph_update",
                "action": step["action"],
                "data": step["data"],
                "timestamp": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(graph_event)}\n\n"
            await asyncio.sleep(2)  # 间隔2秒
        
        # 5. 发送完成信号
        finish_event = {
            "type": "control",
            "status": "finish",
            "stop_reason": "completed",
            "summary": {
                "newNodes": 2,
                "newLinks": 2,
                "totalProcessingTime": "3.5秒"
            }
        }
        yield f"data: {json.dumps(finish_event)}\n\n"
        
        # 6. 流结束
        yield f"data: {json.dumps({'type': 'control', 'status': 'closed'})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

# ==================== 运行应用 ====================
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("EvoMind 后端服务启动中...")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时自动重载
        log_level="info"
    )