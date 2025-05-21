import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from baseline.pipeline import RAGPipeline

def load_test_inputs(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def log_result(result: dict, log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def evaluate_pipeline(data_path, retriever_store, test_input_path, log_path):
    pipeline = RAGPipeline(data_path, retriever_store)
    tests = load_test_inputs(test_input_path)

    passed = 0
    for test in tests:
        result = pipeline.run(test["question"])
        log_result(result, log_path)

        answer = result["generated_answer"]
        expected = test["expected_answer_contains"].lower()

        if expected in answer.lower():
            result_status = "✅ PASS"
            passed += 1
        else:
            result_status = "❌ FAIL"

        print(f"Q: {test['question']}\nExpected: {expected}\nGot: {answer}\n{result_status}\n")

    print(f"\n{passed}/{len(tests)} test cases passed.")

if __name__ == "__main__":
    evaluate_pipeline(
        data_path="baseline/data/tech_facts.txt",
        retriever_store="baseline/retriever_store",
        test_input_path="evaluation/test_inputs.json",
        log_path="evaluation/logs.jsonl"
    )
