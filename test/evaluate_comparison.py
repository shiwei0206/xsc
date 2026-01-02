import asyncio
import json
import os
from core.brain import DualBrain
from openai import OpenAI
from dotenv import load_dotenv

# 加载配置
load_dotenv()


class ComparisonEvaluator:
    def __init__(self):
        # 初始化双脑系统 (Ours)
        self.brain = DualBrain()
        # 初始化纯 API 客户端 (Baseline)
        self.raw_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )

        # 统计数据
        self.stats = {
            "total": 0,
            "baseline_correct": 0,
            "baseline_hallucination": 0,
            "ours_correct": 0,
            "ours_hallucination": 0
        }
        self.results = []

    async def get_baseline_answer(self, question: str) -> str:
        """Baseline: 直接调用 DeepSeek，无背景知识"""
        try:
            response = self.raw_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个助手。请简短回答用户的问题。"},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    async def get_system_answer(self, session_id: str, question: str) -> str:
        """Ours: DualBrain 系统，包含检索 (RAG)"""
        full_answer = ""
        # 使用 brain.think 全流程 (检索->生成)
        async for event in self.brain.think(session_id, question):
            try:
                data = json.loads(event.replace("data: ", ""))
                if data['type'] == 'chunk':
                    full_answer += data['content']
            except:
                pass
        return full_answer

    def check_correctness(self, answer: str, keyword: str) -> str:
        """
        简单判定标准：是否包含预期关键词
        - 包含 -> Correct
        - 不包含 -> 判定为 Hallucination (或者 Unknown，这里简化处理)
        """
        if keyword.lower() in answer.lower():
            return "Correct"
        else:
            return "Unknown"  # 简化：只要不对就当是幻觉/错误

    async def run_experiment(self, data_file="eval_data.json"):
        print("开始对比实验: Baseline (DeepSeek) vs Ours (DualBrain)...")

        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        session_id = "experiment_session"

        # --- 阶段 0: 知识注入 (让 Ours 系统先"学习"这些知识) ---
        print("\n[预处理] 正在将测试集知识注入 DualBrain 图谱...")
        for item in dataset:
            # 模拟用户陈述事实，让后台脑学习
            async for _ in self.brain._slow_brain_learn(session_id, item['text']):
                pass  # 静默写入


        # --- 阶段 1: 开始测试 ---
        print("\n开始问答测试...")
        print(f"{'ID':<4} | {'Question':<20} | {'deepseek':<12} | {'Ours':<12} | {'Keyword'}")
        print("-" * 80)

        for item in dataset:
            q = item['question']
            keyword = item['expected_answer_keyword']
            self.stats["total"] += 1

            # 1. 跑 Baseline
            base_ans = await self.get_baseline_answer(q)
            base_res = self.check_correctness(base_ans, keyword)

            # 2. 跑 Ours
            ours_ans = await self.get_system_answer(session_id, q)
            ours_res = self.check_correctness(ours_ans, keyword)

            # 3. 记录统计
            if base_res == "Correct":
                self.stats["baseline_correct"] += 1
            else:
                self.stats["baseline_hallucination"] += 1

            if ours_res == "Correct":
                self.stats["ours_correct"] += 1
            else:
                self.stats["ours_hallucination"] += 1

            # 打印单行结果
            print(f"{item['id']:<4} | {q[:20]:<20} | {base_res:<12} | {ours_res:<12} | {keyword}")

            self.results.append({
                "id": item['id'],
                "question": q,
                "baseline": {"answer": base_ans[:50] + "...", "result": base_res},
                "ours": {"answer": ours_ans[:50] + "...", "result": ours_res}
            })

        self.print_final_report()
        self.brain.close()

    def print_final_report(self):
        total = self.stats["total"]
        if total == 0: return

        # 计算指标
        base_err = self.stats["baseline_hallucination"]
        ours_err = self.stats["ours_hallucination"]

        # 幻觉率
        base_hallucination_rate = base_err / total
        ours_hallucination_rate = ours_err / total

        # 修正率 (Baseline 错的里面，Ours 对了多少？或者整体错误下降比例)
        # 公式: (对照组错 - 实验组错) / 对照组错
        if base_err > 0:
            correction_rate = (base_err - ours_err) / base_err
        else:
            correction_rate = 0.0  # Baseline 全对，没法修正

        print("\n" + "=" * 60)
        print("最终量化评估报告")
        print("=" * 60)
        print(f"测试样本总数: {total}")
        print("-" * 60)
        print(f"1. 对照组 (Baseline - Pure DeepSeek):")
        print(f"   - 正确数: {self.stats['baseline_correct']}")
        print(f"   - 幻觉数: {base_err}")
        print(f"   - 幻觉率: {base_hallucination_rate:.2%}")
        print("-" * 60)
        print(f"2. 实验组 (Ours - DualBrain):")
        print(f"   - 正确数: {self.stats['ours_correct']}")
        print(f"   - 幻觉数: {ours_err}")
        print(f"   - 幻觉率: {ours_hallucination_rate:.2%}")
        print("-" * 60)
        print(f"核心指标:")
        print(f"   - 幻觉修正率 (Correction Rate): {correction_rate:.2%}")
        print("=" * 60)


if __name__ == "__main__":
    import sys

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    evaluator = ComparisonEvaluator()
    asyncio.run(evaluator.run_experiment())