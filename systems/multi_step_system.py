"""
Multi-step reasoning system.

This system executes prompts that use a three-phase approach:
1. Extract/identify core question
2. Extract relevant information
3. Generate solution and answer

Used for techniques like DUP (Deeply Understanding the Problems) Prompting.
"""

from typing import Tuple, Optional
import time
import json

from systems.base_system import BaseSystem
from core.llm_manager import OllamaManager
from core.types import GSM8KExample, ComparisonResult
from prompts.base_prompt import BasePrompt


class MultiStepSystem(BaseSystem):
    """
    Multi-step (three-step) reasoning system.
    
    The system:
    1. Calls the LLM to extract core question
    2. Calls the LLM to extract problem-solving information
    3. Calls the LLM to generate solution using extracted info
    4. Extracts the numerical answer from final solution
    5. Retries on failures
    """
    
    def __init__(self,
                 llm_manager: OllamaManager,
                 prompt: BasePrompt,
                 temperature: float = 0.0,
                 self_consistency: int = 1,
                 max_retries: int = 2):
        """
        Initialize multi-step system.
        
        Args:
            llm_manager: Manager for LLM API calls
            prompt: Prompting strategy to use (must support three-step format)
            temperature: Sampling temperature
            self_consistency: Number of samples for self-consistency voting
            max_retries: Maximum retry attempts on extraction failure
        """
        super().__init__(llm_manager, prompt, temperature, self_consistency, max_retries)
    
    def solve(self, question: str) -> Tuple[str, Optional[float], int, int, str]:
        """
        Solve a question using three-step reasoning with retry logic.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            Tuple of:
            - solution_response: The final solution response text
            - final_num: Extracted numerical answer (or None)
            - total_tokens: Total tokens used
            - retry_count: Number of retries performed
            - retry_reason: Reason for retries
        """
        retry_count = 0
        retry_reason = ""
        plan_response = ""
        info_response = ""
        solution_response = ""
        final_num = None
        total_tokens = 0
        
        for attempt in range(self.max_retries + 1):
            # Increase temperature slightly on retries
            current_temp = self.retry_handler.get_retry_temperature(
                self.temperature, attempt
            )
            
            # Step 1: Extract core question
            plan_prompt = self.prompt.format_plan_prompt(question)
            plan_response, tokens1 = self.llm.call_api(plan_prompt, temperature=current_temp)
            total_tokens += tokens1
            
            # Step 2: Extract problem-solving information
            info_prompt = self.prompt.format_info_prompt(question, plan_response)
            info_response, tokens2 = self.llm.call_api(info_prompt, temperature=current_temp)
            total_tokens += tokens2
            
            # Step 3: Generate solution using extracted information
            solve_prompt = self.prompt.format_solve_prompt(question, plan_response, info_response)
            solution_response, tokens3 = self.llm.call_api(solve_prompt, temperature=current_temp)
            total_tokens += tokens3
            
            # Extract numerical answer from final solution
            final_num = self.extractor.extract_number(solution_response)
            
            # Check if retry is needed
            should_retry, reason = self.retry_handler.should_retry(
                solution_response, final_num, attempt, self.max_retries
            )
            
            if should_retry:
                retry_count += 1
                retry_reason = reason
                # Small delay before retry
                time.sleep(0.5)
                continue
            
            # Success - exit retry loop
            break
        
        # Store intermediate responses for evaluate() to access
        self._last_plan_response = plan_response
        self._last_info_response = info_response
        
        return solution_response, final_num, total_tokens, retry_count, retry_reason
    
    def evaluate(self, example: GSM8KExample) -> ComparisonResult:
        """
        Evaluate a single GSM8K example (override to populate all response fields).
        
        Args:
            example: GSM8K problem instance
            
        Returns:
            ComparisonResult with evaluation metrics and all intermediate responses
        """
        t0 = time.time()
        sc_mode = "enabled" if self.self_consistency > 1 else "disabled"
        
        # Solve with or without self-consistency
        if self.self_consistency <= 1:
            response, final_num, tokens, retry_count, retry_reason = self.solve(example.question)
            all_responses = []
            numerical_list = [final_num]
            plan_response = getattr(self, '_last_plan_response', '')
            info_response = getattr(self, '_last_info_response', '')
        else:
            response, all_responses, tokens, numerical_list, retry_count, retry_reason = \
                self.solve_with_self_consistency(example.question)
            # Re-extract from chosen response for consistency
            final_num = self.extractor.extract_number(response)
            plan_response = getattr(self, '_last_plan_response', '')
            info_response = getattr(self, '_last_info_response', '')
        
        processing_time = time.time() - t0
        
        # Check correctness
        is_correct = final_num is not None and abs(final_num - example.numerical_answer) < 0.01
        
        return ComparisonResult(
            original_question=example.question,
            ground_truth_answer=example.answer,
            ground_truth_numerical=example.numerical_answer,
            main_response=response,  # Step 3: Final solution
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
            retry_reason=retry_reason,
            extraction_response="",  # Not used in three-step
            plan_response=plan_response,  # Step 1: Core question
            info_response=info_response   # Step 2: Problem-solving info
        )