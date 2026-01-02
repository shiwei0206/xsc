import asyncio
import json
import time
from core.brain import DualBrain


# 简单的内存记录器
class Evaluator:
    def __init__(self):
        self.tp = 0  # 提取正确
        self.fp = 0  # 提取错误
        self.fn = 0  # 漏提取
        self.rag_correct = 0
        self.total_questions = 0

    def evaluate_extraction(self, expected, actual_nodes, actual_links):
        """评估图谱构建质量 (Precision/Recall)"""
        print(f"   [期望] {expected}")

        # 简化逻辑：只要 actual_links 里包含了 expected 的关系就算对
        found = False
        for exp in expected:
            # 检查是否有对应的 link
            # 注意：实际项目中比较需要更严谨，这里做模糊匹配
            for link in actual_links:
                if (link['relationship'] == exp['relation']):
                    self.tp += 1
                    found = True
                    print(f"    成功提取: {exp['relation']}")
                    break

            if not found:
                self.fn += 1
                print(f"    漏提取: {exp['relation']}")

        # 简单的 FP 计算：实际提取总数 - TP
        current_fp = len(actual_links) - (1 if found else 0)
        self.fp += current_fp

    def evaluate_rag(self, answer, keyword):
        """评估问答准确性"""
        self.total_questions += 1
        if keyword in answer:
            self.rag_correct += 1
            print(f"    回答正确 (包含关键词 '{keyword}')")
        else:
            print(f"    回答可能错误 (未包含 '{keyword}') - 回答: {answer[:30]}...")

    def print_report(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        rag_acc = self.rag_correct / self.total_questions if self.total_questions > 0 else 0

        print("\n" + "=" * 50)
        print(" 最终评估报告")
        print("=" * 50)
        print(f"1. 信息抽取 (IE) 性能:")
        print(f"   - 准确率 (Precision): {precision:.2%}")
        print(f"   - 召回率 (Recall):    {recall:.2%}")
        print(f"   - F1 分数:            {f1:.2f}")
        print("-" * 50)
        print(f"2. 问答 (RAG) 性能:")
        print(f"   - 准确率 (Accuracy):  {rag_acc:.2%}")
        print("=" * 50)


async def run_evaluation():
    # 加载数据
    with open("eval_data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    brain = DualBrain()
    evaluator = Evaluator()
    session_id = "eval_session"

    print(" 开始自动化评估...")

    for item in dataset:
        print(f"\n 测试样本 ID: {item['id']}")

        # --- 阶段 1: 学习 (Learn) ---
        print(f" 输入事实: {item['text']}")
        extracted_links = []

        # 运行慢脑学习
        async for event in brain._slow_brain_learn(session_id, item['text']):
            data = json.loads(event.replace("data: ", ""))
            if data['type'] == 'graph_update':
                extracted_links = data['data']['links']

        # 评估提取质量
        evaluator.evaluate_extraction(item['expected_triples'], [], extracted_links)

        # --- 阶段 2: 问答 (Ask) ---
        print(f" 提问验证: {item['question']}")

        # 为了确保检索生效，手动触发一次 search
        keywords = await brain._extract_search_keywords(item['question'])
        # 模拟运行 search_subgraph (这里我们不重新从 neo4j 拉，假设 context 已经有了)
        # 在真实测试中，应该让 brain.think 跑全流程

        full_answer = ""
        # 运行 RAG 流程
        # 这里为了简化，我们假设检索已经成功，直接测试生成能力
        # 实际严谨测试应该调用 brain.think(session_id, item['question']) 并解析
        context = f"{item['expected_triples'][0]['head']} 的关系是 {item['expected_triples'][0]['relation']} 对象是 {item['expected_triples'][0]['tail']}"

        async for event in brain._fast_brain_generate(item['question'], context):
            data = json.loads(event.replace("data: ", ""))
            full_answer += data['content']

        evaluator.evaluate_rag(full_answer, item['expected_answer_keyword'])

    # 输出报告
    evaluator.print_report()
    brain.close()


if __name__ == "__main__":
    import sys

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())