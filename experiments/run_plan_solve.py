#!/usr/bin/env python3
"""
Plan-and-Solve Experiment Runner

This script evaluates the Plan-and-Solve prompting technique on GSM8K.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import OllamaManager, GSM8KDataLoader, GSM8KExample, ComparisonResult
from prompts.plan_solve import PlanSolvePrompt
from systems.single_step_system import SingleStepSystem
from utils import ResultFormatter


def run_experiment(
    examples: list,
    ollama_url: str,
    model: str,
    temperature: float,
    self_consistency: int,
    max_retries: int,
    output_dir: Path,
    seed: int = 42
) -> list:
    """
    Run Plan-and-Solve experiment on GSM8K examples.
    
    Args:
        examples: List of GSM8K problems to evaluate
        ollama_url: Ollama server URL
        model: Model name
        temperature: Sampling temperature
        self_consistency: Self-consistency count
        max_retries: Maximum retries
        output_dir: Directory to save results
        seed: Random seed used for sampling
        
    Returns:
        List of evaluation results
    """
    # Initialize components
    llm = OllamaManager(base_url=ollama_url, model=model)
    prompt = PlanSolvePrompt()
    system = SingleStepSystem(
        llm_manager=llm,
        prompt=prompt,
        temperature=temperature,
        self_consistency=self_consistency,
        max_retries=max_retries
    )
    
    # Print experiment info
    print("=" * 70)
    print("Plan-and-Solve Evaluation")
    print("=" * 70)
    print(f"Model                     : {model}")
    print(f"Ollama URL                : {ollama_url}")
    print(f"Temperature               : {temperature}")
    print(f"Self-Consistency          : {self_consistency}")
    print(f"Max Retries               : {max_retries}")
    print(f"Seed                      : {seed}")
    print(f"Total Samples             : {len(examples)}")
    print("=" * 70)
    
    # Evaluate examples
    results = []
    for i, example in enumerate(examples, 1):
        try:
            result = system.evaluate(example)
            results.append(result)
            
            # Print progress
            ResultFormatter.print_progress(
                current=i,
                total=len(examples),
                question=example.question,
                is_correct=result.is_correct,
                predicted=result.final_numerical,
                ground_truth=result.ground_truth_numerical,
                retry_count=result.retry_count
            )
            
        except Exception as e:
            print(f"[Error] Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results and print summary
    ResultFormatter.save_to_csv(
        results=results,
        output_dir=output_dir,
        technique_name=prompt.name,
        model_name=model,
        temperature=temperature,
        self_consistency=self_consistency,
        max_retries=max_retries,
        seed=seed
    )
    
    ResultFormatter.print_summary(
        results=results,
        technique_name=prompt.display_name,
        model_name=model,
        temperature=temperature,
        self_consistency=self_consistency,
        max_retries=max_retries
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Plan-and-Solve on GSM8K"
    )
    
    # Data arguments
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Model arguments
    parser.add_argument("--ollama-url", type=str, 
                       default="http://localhost:11434",
                       help="Ollama server URL")
    parser.add_argument("--model", type=str, 
                       default="deepseek-r1:1.5b",
                       help="Ollama model name")
    parser.add_argument("--temp", type=float, default=0.0,
                       help="Sampling temperature")
    
    # Technique arguments
    parser.add_argument("--sc", type=int, default=1,
                       help="Self-consistency voting count (1 = disabled)")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Maximum retry attempts on extraction failure")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, 
                       default="results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading GSM8K dataset...")
    loader = GSM8KDataLoader()
    examples = loader.load_sample_data(
        n_samples=args.samples,
        split=args.split,
        seed=args.seed
    )
    print(f"Loaded {len(examples)} examples\n")
    
    # Run experiment
    output_dir = Path(args.output_dir)
    results = run_experiment(
        examples=examples,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temp,
        self_consistency=args.sc,
        max_retries=args.max_retries,
        output_dir=output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()