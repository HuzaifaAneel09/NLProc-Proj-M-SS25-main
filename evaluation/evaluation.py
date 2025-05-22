import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_question", type=str, help="Ask a custom question")
    args = parser.parse_args()

    if args.custom_question:
        pipeline = RAGPipeline(
            data_path="baseline/data/tech_facts.txt",
            retriever_store="baseline/retriever_store"
        )
        result = pipeline.run(args.custom_question)

        # Log the result
        log_result(result, "evaluation/logs.jsonl")

        print("\n--- Retrieved Context ---")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"[{i}] {chunk}")

        print("\n--- Prompt ---")
        print(result['prompt'])

        print("\n--- Generated Answer ---")
        print(result['generated_answer'])
    else:
        evaluate_pipeline(
            data_path="baseline/data/tech_facts.txt",
            retriever_store="baseline/retriever_store",
            test_input_path="evaluation/test_inputs.json",
            log_path="evaluation/logs.jsonl"
        )
