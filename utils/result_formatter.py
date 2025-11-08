"""
Result formatting and summary statistics for GSM8K evaluation.

This module handles saving results to CSV and generating summary statistics
for experimental runs.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import asdict

from core.types import ComparisonResult


class ResultFormatter:
    """
    Handles formatting and saving of evaluation results.
    
    Provides consistent CSV output format and summary statistics
    across all prompting techniques.
    """
    
    @staticmethod
    def save_to_csv(results: List[ComparisonResult], 
                   output_dir: Path,
                   technique_name: str,
                   model_name: str,
                   temperature: float,
                   self_consistency: int,
                   max_retries: int,
                   seed: int = 42) -> Path:
        """
        Save results to a timestamped CSV file organized by model and technique.
        
        Directory structure:
        results/
        └── {model_name}/
            └── {technique_name}/
                └── {technique}_{model}_temp{temp}_sc{sc}_retry{retry}_seed{seed}_{timestamp}.csv
        
        Args:
            results: List of ComparisonResult instances
            output_dir: Base directory to save results
            technique_name: Name of the prompting technique
            model_name: Name of the model used
            temperature: Temperature setting
            self_consistency: Self-consistency count
            max_retries: Maximum retries setting
            seed: Random seed used for sampling
            
        Returns:
            Path to the saved CSV file
        """
        if not results:
            print("No results to save.")
            return None
        
        # Create organized directory structure: results/{model}/{technique}/
        model_safe = model_name.replace(":", "_").replace("/", "_")
        technique_dir = output_dir / model_safe / technique_name
        technique_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"{technique_name}_{model_safe}_"
                   f"temp{temperature}_sc{self_consistency}_"
                   f"retry{max_retries}_seed{seed}_{timestamp}.csv")
        filepath = technique_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Determine which columns to keep based on technique type
        # Check if any result has non-empty optional columns
        has_extraction = any(r.extraction_response != "" for r in results)
        has_plan = any(r.plan_response != "" for r in results)
        has_info = any(r.info_response != "" for r in results)
        
        # Define column order based on what's used
        base_columns = [
            'original_question',
            'ground_truth_answer', 
            'ground_truth_numerical',
            'main_response'
        ]
        
        # Add intermediate response columns in proper order if used
        if has_extraction:
            base_columns.append('extraction_response')
        if has_plan:
            base_columns.append('plan_response')
        if has_info:
            base_columns.append('info_response')
        
        # Add remaining columns
        remaining_columns = [
            'final_numerical',
            'is_correct',
            'processing_time',
            'tokens_used',
            'temperature',
            'self_consistency',
            'sc_mode',
            'sc_count',
            'sc_all_answers',
            'sc_numerical_answers',
            'model_name',
            'retry_count',
            'retry_reason'
        ]
        
        base_columns.extend(remaining_columns)
        
        # Reorder and filter columns
        df = df[base_columns]
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        print(f"\nResults saved to {filepath}")
        return filepath

    @staticmethod
    def print_summary(results: List[ComparisonResult], 
                     technique_name: str,
                     model_name: str,
                     temperature: float,
                     self_consistency: int,
                     max_retries: int):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            results: List of ComparisonResult instances
            technique_name: Name of the prompting technique
            model_name: Name of the model used
            temperature: Temperature setting
            self_consistency: Self-consistency count
            max_retries: Maximum retries setting
        """
        if not results:
            print("No results to summarize.")
            return
        
        total = len(results)
        correct = sum(r.is_correct for r in results)
        accuracy = correct / total if total > 0 else 0
        avg_time = sum(r.processing_time for r in results) / total
        avg_tokens = sum(r.tokens_used for r in results) / total
        
        # Retry statistics
        retry_count = sum(r.retry_count for r in results)
        questions_retried = sum(1 for r in results if r.retry_count > 0)
        
        # Retry success rate
        retried_results = [r for r in results if r.retry_count > 0]
        if retried_results:
            retry_success = sum(1 for r in retried_results if r.is_correct) / len(retried_results)
        else:
            retry_success = 0.0
        
        # Print formatted summary
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"Model                     : {model_name}")
        print(f"Technique                 : {technique_name}")
        print(f"Total Samples             : {total}")
        print(f"Correct                   : {correct}")
        print(f"Accuracy                  : {accuracy:.2%}")
        print(f"Temperature               : {temperature}")
        print(f"Self-Consistency          : {self_consistency}")
        print(f"Max Retries               : {max_retries}")
        print(f"Avg Time (sec)            : {avg_time:.2f}")
        print(f"Avg Tokens Used           : {avg_tokens:.1f}")
        
        if max_retries > 0:
            print(f"\nRetry Statistics:")
            print(f"Questions Retried         : {questions_retried}/{total} "
                  f"({questions_retried/total*100:.1f}%)")
            print(f"Total Retry Attempts      : {retry_count}")
            if retried_results:
                print(f"Retry Success Rate        : {retry_success:.2%}")
        
        print("=" * 70)
    
    @staticmethod
    def print_progress(current: int, total: int, question: str, 
                      is_correct: bool, predicted: Optional[float], 
                      ground_truth: float, retry_count: int = 0):
        """
        Print progress during evaluation.
        
        Args:
            current: Current question number
            total: Total number of questions
            question: Question text (will be truncated)
            is_correct: Whether prediction was correct
            predicted: Predicted answer
            ground_truth: Ground truth answer
            retry_count: Number of retries for this question
        """
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        question_preview = question[:80] + "..." if len(question) > 80 else question
        
        print(f"\nEvaluating {current}/{total}: {question_preview}")
        print(f"    {status} - Predicted: {predicted}, Ground Truth: {ground_truth}")
        
        if retry_count > 0:
            print(f"    Retries: {retry_count}")