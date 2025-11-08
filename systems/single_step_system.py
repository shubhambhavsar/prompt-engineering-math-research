"""
Single-step reasoning system.

This system executes prompts that generate both reasoning and answer in one pass,
such as Zero-Shot CoT and basic Plan-and-Solve.
"""

from typing import Tuple, Optional
import time

from systems.base_system import BaseSystem
from core.llm_manager import OllamaManager
from prompts.base_prompt import BasePrompt


class SingleStepSystem(BaseSystem):
    """
    Single-step reasoning system for prompts that solve in one pass.
    
    Used for techniques like:
    - Zero-Shot Chain-of-Thought
    - Basic Plan-and-Solve
    - Plan-and-Solve Plus
    
    The system:
    1. Formats the question using the prompt strategy
    2. Calls the LLM once to generate reasoning and answer
    3. Extracts the numerical answer
    4. Retries on failures (incomplete response, extraction failure, etc.)
    """
    
    def __init__(self,
                 llm_manager: OllamaManager,
                 prompt: BasePrompt,
                 temperature: float = 0.0,
                 self_consistency: int = 1,
                 max_retries: int = 2):
        """
        Initialize single-step system.
        
        Args:
            llm_manager: Manager for LLM API calls
            prompt: Prompting strategy to use
            temperature: Sampling temperature
            self_consistency: Number of samples for self-consistency voting
            max_retries: Maximum retry attempts on extraction failure
        """
        super().__init__(llm_manager, prompt, temperature, self_consistency, max_retries)
    
    def solve(self, question: str) -> Tuple[str, Optional[float], int, int, str]:
        """
        Solve a question using single-step reasoning with retry logic.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            Tuple of:
            - response: The model's response text
            - final_num: Extracted numerical answer (or None)
            - total_tokens: Total tokens used
            - retry_count: Number of retries performed
            - retry_reason: Reason for retries
        """
        retry_count = 0
        retry_reason = ""
        response = ""
        final_num = None
        total_tokens = 0
        
        for attempt in range(self.max_retries + 1):
            # Increase temperature slightly on retries
            current_temp = self.retry_handler.get_retry_temperature(
                self.temperature, attempt
            )
            
            # Format prompt and call LLM
            prompt_text = self.prompt.format_prompt(question)
            response, tokens = self.llm.call_api(prompt_text, temperature=current_temp)
            total_tokens += tokens
            
            # Extract numerical answer
            final_num = self.extractor.extract_number(response)
            
            # Check if retry is needed
            should_retry, reason = self.retry_handler.should_retry(
                response, final_num, attempt, self.max_retries
            )
            
            if should_retry:
                retry_count += 1
                retry_reason = reason
                # Small delay before retry
                time.sleep(0.5)
                continue
            
            # Success - exit retry loop
            break
        
        return response, final_num, total_tokens, retry_count, retry_reason