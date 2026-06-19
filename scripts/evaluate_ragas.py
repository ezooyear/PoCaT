"""
Ragas RAG Evaluation Script for PoCaT.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from config.settings import get_llm
from db.vectorstore import search_products, _get_embeddings
from graph.builder import build_graph
from agents.product.tools import reformulate_query


def main():
    print("=" * 60)
    print("PoCaT RAGAS Evaluation Pipeline")
    print("=" * 60)

    # 1. Load Golden Dataset
    dataset_path = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"
    if not dataset_path.exists():
        print(f"❌ Golden dataset not found at: {dataset_path}")
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"📋 Loaded {len(golden_data)} test cases from Golden Dataset.")

    # 2. Build LangGraph and Load Components
    graph = build_graph()
    judge_llm = get_llm()
    embeddings = _get_embeddings()

    eval_samples = []

    # 3. Execute RAG & Agent Loop to gather answers and contexts
    print("\n🔄 Running evaluation cases through the agent graph...")
    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"   [{i}/{len(golden_data)}] Question: {question[:40]}...")

        # A. Retrieve contexts (Parent Chunks) with Query Reformulation and Dynamic K
        is_comparative = any(keyword in question for keyword in ["비교", "차이", "모두", "목록", "공통", "다른점"])
        target_k = 6 if is_comparative else 4
        
        reformed_query = reformulate_query(question)
        retrieved_docs = search_products(reformed_query, k=target_k)
        contexts = [doc.page_content for doc in retrieved_docs]

        # B. Run Graph to get Agent's Final Answer
        try:
            result = graph.invoke({
                "messages": [("user", question)],
                "next": "",
                "member_id": None,
                "context": None,
                "plan": [],
                "current_step": 0,
                "agent_outputs": {},
            })
            
            # Extract final assistant message content
            ai_message = result["messages"][-1]
            answer = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        except Exception as e:
            print(f"      ⚠️ Error during graph invocation: {e}")
            answer = f"Error occurred: {e}"

        eval_samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": ground_truth,
        })

    # 4. Format Dataset for Ragas
    # Ragas expects: question (str), contexts (list[str]), answer (str), ground_truth (str)
    hf_dataset = Dataset.from_list(eval_samples)
    dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    # 5. Initialize Ragas Metrics with Judge LLM and Embeddings
    # This prevents Ragas from trying to access OpenAI API directly if not configured
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # 6. Execute Ragas Evaluation
    print("\n📊 Evaluating with Ragas...")
    try:
        score_result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        # 7. Print and Save Results
        print("\n" + "=" * 60)
        print("RAGAS Evaluation Scores")
        print("=" * 60)
        print(score_result)
        print("=" * 60)

        # Save report as CSV
        report_dir = PROJECT_ROOT / "data"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "ragas_eval_results.csv"
        
        df = score_result.to_pandas()
        df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"💾 Detailed report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Ragas evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
