"""
Base abstract class for reasoning systems.

This module defines the interface for systems that execute prompting strategies
to solve mathematical reasoning problems.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import time
import json

from core.llm_manager import OllamaManager
from core.answer_extraction import AnswerExtractor
from core.types import GSM8KExample, ComparisonResult
from prompts.base_prompt import BasePrompt
from utils.retry_logic import RetryHandler
from utils.self_consistency import SelfConsistency


class BaseSystem(ABC):
    """
    Abstract base class for reasoning systems.
    
    A reasoning system combines:
    1. A prompting strategy (what to ask)
    2. An execution strategy (how to ask it - single-step, multi-step, etc.)
    3. Retry logic and self-consistency voting
    """
    
    def __init__(self, 
                 llm_manager: OllamaManager,
                 prompt: BasePrompt,
                 temperature: float = 0.0,
                 self_consistency: int = 1,
                 max_retries: int = 2):
        """
        Initialize the reasoning system.
        
        Args:
            llm_manager: Manager for LLM API calls
            prompt: Prompting strategy to use
            temperature: Sampling temperature
            self_consistency: Number of samples for self-consistency voting (1 = disabled)
            max_retries: Maximum retry attempts on extraction failure
        """
        self.llm = llm_manager
        self.prompt = prompt
        self.temperature = temperature
        self.self_consistency = self_consistency
        self.max_retries = max_retries
        self.extractor = AnswerExtractor()
        self.retry_handler = RetryHandler()
    
    @abstractmethod
    def solve(self, question: str) -> Tuple[str, Optional[float], int, int, str]:
        """
        Solve a question using this reasoning system.
        
        This is the core method that must be implemented by each system.
        It defines how the prompting strategy is executed.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            Tuple of:
            - response: The main response text
            - final_num: Extracted numerical answer (or None)
            - tokens: Total tokens used
            - retry_count: Number of retries performed
            - retry_reason: Reason for retries (comma-separated if multiple)
        """
        pass
    
    def solve_with_self_consistency(self, question: str) -> Tuple[str, List[str], int, List[Optional[float]], int, str]:
        """
        Solve using self-consistency voting.
        
        Calls solve() multiple times and uses majority voting to select
        the most consistent answer.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            Tuple of:
            - winning_response: Response text of the winning answer
            - all_responses: All response texts generated
            - total_tokens: Sum of tokens across all samples
            - numerical_answers: All extracted numerical answers
            - max_retry_count: Maximum retry count across samples
            - combined_retry_reason: Combined retry reasons
        """
        all_responses = []
        numerical_answers = []
        total_tokens = 0
        max_retry_count = 0
        retry_reasons = []
        
        for i in range(self.self_consistency):
            response, final_num, tokens, retry_count, retry_reason = self.solve(question)
            
            all_responses.append(response)
            numerical_answers.append(final_num)
            total_tokens += tokens
            max_retry_count = max(max_retry_count, retry_count)
            
            if retry_reason:
                retry_reasons.append(retry_reason)
        
        if not all_responses:
            return "", [], 0, [], 0, ""
        
        # Perform majority voting
        winning_idx, winning_val, vote_count = SelfConsistency.majority_vote(
            numerical_answers, all_responses
        )
        
        # If no valid answers, return first response
        if not numerical_answers or all(num is None for num in numerical_answers):
            combined_retry_reason = ",".join(set(retry_reasons)) if retry_reasons else ""
            return all_responses[0], all_responses, total_tokens, numerical_answers, max_retry_count, combined_retry_reason
        
        combined_retry_reason = ",".join(set(retry_reasons)) if retry_reasons else ""
        return (all_responses[winning_idx], all_responses, total_tokens, 
                numerical_answers, max_retry_count, combined_retry_reason)
    
    def evaluate(self, example: GSM8KExample) -> ComparisonResult:
        """
        Evaluate a single GSM8K example.
        
        Args:
            example: GSM8K problem instance
            
        Returns:
            ComparisonResult with evaluation metrics
        """
        t0 = time.time()
        sc_mode = "enabled" if self.self_consistency > 1 else "disabled"
        
        # Solve with or without self-consistency
        if self.self_consistency <= 1:
            response, final_num, tokens, retry_count, retry_reason = self.solve(example.question)
            all_responses = []
            numerical_list = [final_num]
        else:
            response, all_responses, tokens, numerical_list, retry_count, retry_reason = \
                self.solve_with_self_consistency(example.question)
            # Re-extract from chosen response for consistency
            final_num = self.extractor.extract_number(response)
        
        processing_time = time.time() - t0
        
        # Check correctness
        is_correct = final_num is not None and abs(final_num - example.numerical_answer) < 0.01
        
        return ComparisonResult(
            original_question=example.question,
            ground_truth_answer=example.answer,
            ground_truth_numerical=example.numerical_answer,
            main_response=response,
            final_numerical=final_num,
            is_correct=is_correct,
            processing_time=processing_time,
            tokens_used=tokens,
            temperature=self.temperature,
            self_consistency=self.self_consistency,
            sc_mode=sc_mode,
            sc_count=len(all_responses) if all_responses else 1,
            sc_all_answers=json.dumps(all_responses) if all_responses else json.dumps([response]),
            sc_numerical_answers=json.dumps(numerical_list),
            model_name=self.llm.get_model(),
            retry_count=retry_count,
            retry_reason=retry_reason
        )