#!/usr/bin/env python3
"""
gsm8k_dup_enhanced_v3.py

DUP GSM8K evaluation pipeline with:
1. Reproducible sampling via --seed
2. Self-consistency voting + completions saved to CSV
3. Accurate token usage tracking per example
"""

import os
import json
import random
import time
import re
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from fractions import Fraction
from groq import Groq
from datetime import datetime
from collections import Counter

load_dotenv()

@dataclass
class GSM8KExample:
    question: str
    answer: str
    numerical_answer: float

@dataclass
class DUPResult:
    original_question: str
    ground_truth_answer: str
    ground_truth_numerical: float
    core_question: str
    problem_solving_info: str
    final_response: str
    final_numerical_answer: Optional[float]
    zero_shot_cot_response: str
    zero_shot_cot_numerical: Optional[float]
    dup_correct: bool
    zero_shot_cot_correct: bool
    processing_time: float
    tokens_used: int
    temperature: float
    self_consistency: int
    sc_mode: str
    sc_count: int
    sc_majority_answer: str
    sc_all_answers: str  # JSON string for portability

class GSM8KDataLoader:
    def __init__(self):
        self.dataset = None

    def load_sample_data(self, n_samples: int = 50, split: str = "test", seed: int = 42) -> List[GSM8KExample]:
        self.dataset = load_dataset("openai/gsm8k", "main")
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not available. Choose from {list(self.dataset.keys())}")
        data = self.dataset[split]
        random.seed(seed)
        sample_indices = random.sample(range(len(data)), min(n_samples, len(data)))

        examples = []
        for idx in sample_indices:
            item = data[idx]
            num_answer = self._extract_numerical_answer(item["answer"])
            examples.append(GSM8KExample(item["question"], item["answer"], num_answer))

        return examples

    def load_from_jsonl(self, path: str) -> List[GSM8KExample]:
        with open(path, "r") as f:
            lines = f.readlines()
        examples = []
        for line in lines:
            obj = json.loads(line)
            examples.append(GSM8KExample(obj["question"], obj.get("answer", ""), obj.get("numerical_answer", 0.0)))
        return examples

    def _extract_numerical_answer(self, answer_text: str) -> float:
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
        if match:
            return float(match.group(1))
        fallback = re.findall(r'-?\d+\.?\d*', answer_text)
        return float(fallback[-1]) if fallback else 0.0

class DUPSystem:
    def __init__(self, api_key: str, temperature: float = 0.0, self_consistency: int = 1):
        self.client = Groq(api_key=api_key)
        self.temperature = temperature
        self.self_consistency = self_consistency

        self.prompts = {
            "stage1": "Please extract core question, only the most comprehensive and detailed one!",
            "stage2": "Note: Please extract the problem-solving information related to the core question ({core_question}), only extract the most useful information, list them one by one!",
            "stage3": (
                "Hint: {problem_solving_info}\n"
                "{core_question}\n\n"
                "Please understand the hint and question carefully.\n"
                "Solve the problem step by step, clearly showing all calculations like this: a * b = <<a*b=result>>result.\n"
                "After solving, write the final numeric answer at the end in this format:\n"
                "#### <final_answer>\n"
                "Do not include any units or explanation after that line."
            ),
            "cot": "Let's think step by step."
        }

    def _call_llm(self, prompt: str) -> (str, int):
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return text, tokens

    def _self_consistent(self, prompt: str) -> (str, List[str], int):
        completions = []
        total_tokens = 0
        for _ in range(self.self_consistency):
            try:
                response, tokens = self._call_llm(prompt)
                completions.append(response)
                total_tokens += tokens
            except:
                continue
        if not completions:
            return "", [], 0
        majority = Counter(completions).most_common(1)[0][0]
        return majority, completions, total_tokens

    def run_single_stage(self, prompt: str) -> (str, int, List[str]):
        if self.self_consistency <= 1:
            output, tokens = self._call_llm(prompt)
            return output, tokens, []
        else:
            output, all_answers, tokens = self._self_consistent(prompt)
            return output, tokens, all_answers

    def _extract_numeric_from_llm(self, text: str) -> Optional[float]:
        prompt = (
            "Extract the single FINAL numeric answer from the text below.\n"
            "Respond in strict JSON format: {\"value\": <number or null>}.\n"
            "Convert fractions to decimals. Ignore symbols and commas.\n\n"
            f"Text:\n{text}"
        )
        try:
            result, _ = self._call_llm(prompt)
            if result.startswith("```"):
                result = "\n".join(result.strip("```").splitlines()[1:-1])
            obj = json.loads(result)
            val = obj.get("value")
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return float(Fraction(val.replace(",", "").strip()))
        except:
            return None

    def evaluate(self, example: GSM8KExample) -> DUPResult:
        t0 = time.time()
        sc_mode = "enabled" if self.self_consistency > 1 else "disabled"

        stage1, tok1, _ = self.run_single_stage(f"{example.question}\n{self.prompts['stage1']}")
        stage2, tok2, _ = self.run_single_stage(f"{example.question}\n{self.prompts['stage2'].format(core_question=stage1)}")
        stage3, tok3, all_sc = self.run_single_stage(f"{example.question}\n{self.prompts['stage3'].format(core_question=stage1, problem_solving_info=stage2)}")
        cot, tok4, _ = self.run_single_stage(f"{example.question}\n{self.prompts['cot']}")

        dup_num = self._extract_numeric_from_llm(stage3)
        cot_num = self._extract_numeric_from_llm(cot)

        tokens_used = tok1 + tok2 + tok3 + tok4
        processing_time = time.time() - t0
        dup_correct = dup_num is not None and abs(dup_num - example.numerical_answer) < 0.01
        cot_correct = cot_num is not None and abs(cot_num - example.numerical_answer) < 0.01

        return DUPResult(
            original_question=example.question,
            ground_truth_answer=example.answer,
            ground_truth_numerical=example.numerical_answer,
            core_question=stage1,
            problem_solving_info=stage2,
            final_response=stage3,
            final_numerical_answer=dup_num,
            zero_shot_cot_response=cot,
            zero_shot_cot_numerical=cot_num,
            dup_correct=dup_correct,
            zero_shot_cot_correct=cot_correct,
            processing_time=processing_time,
            tokens_used=tokens_used,
            temperature=self.temperature,
            self_consistency=self.self_consistency,
            sc_mode=sc_mode,
            sc_count=len(all_sc),
            sc_majority_answer=stage3,
            sc_all_answers=json.dumps(all_sc)
        )

class GSM8KEvaluator:
    def __init__(self, temperature: float, self_consistency: int):
        self.results = []
        self.temp = temperature
        self.sc = self_consistency

    def run(self, examples: List[GSM8KExample]):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        system = DUPSystem(api_key=key, temperature=self.temp, self_consistency=self.sc)
        for i, ex in enumerate(examples, 1):
            try:
                print(f"Evaluating {i}/{len(examples)}: {ex.question[:80]}...")
                result = system.evaluate(ex)
                self.results.append(result)
            except Exception as e:
                print(f"Error on example {i}: {e}")
        return self.results

    def save_to_csv(self):
        if not self.results:
            print("No results to save.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gsm8k_results_temp{self.temp}_sc{self.sc}_{ts}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def summary(self):
        if not self.results:
            print("No results to summarize.")
            return
        total = len(self.results)
        dup_acc = sum(r.dup_correct for r in self.results) / total
        cot_acc = sum(r.zero_shot_cot_correct for r in self.results) / total
        avg_time = sum(r.processing_time for r in self.results) / total

        print("\nEvaluation Summary")
        print("=" * 50)
        print(f"Total Samples     : {total}")
        print(f"DUP Accuracy      : {dup_acc:.2%}")
        print(f"Zero-shot CoT Acc : {cot_acc:.2%}")
        print(f"Improvement       : {dup_acc - cot_acc:.2%}")
        print(f"Avg Time (sec)    : {avg_time:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--file", type=str, help="Path to JSONL file with questions")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--sc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = GSM8KDataLoader()
    if args.file:
        examples = loader.load_from_jsonl(args.file)
    else:
        examples = loader.load_sample_data(n_samples=args.samples, split=args.split, seed=args.seed)

    evaluator = GSM8KEvaluator(temperature=args.temp, self_consistency=args.sc)
    evaluator.run(examples)
    evaluator.save_to_csv()
    evaluator.summary()

if __name__ == "__main__":
    main()
