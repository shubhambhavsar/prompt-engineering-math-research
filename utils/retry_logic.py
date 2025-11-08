"""
Unified retry logic for handling incomplete or problematic model responses.

This module consolidates the retry mechanisms used across all prompting
techniques to handle extraction failures, incomplete responses, and
formula-only outputs.
"""

from typing import Optional, Tuple
from core.answer_extraction import AnswerExtractor


class RetryHandler:
    """
    Unified retry logic for all prompting strategies.
    
    All prompting techniques share common failure modes:
    - Incomplete responses (cut off mid-calculation)
    - Formula-only responses (no final numerical answer)
    - Extraction failures (answer present but not extractable)
    
    This handler provides a consistent interface for detecting these
    failures and determining retry strategies.
    """
    
    @staticmethod
    def should_retry(response: str, extracted_num: Optional[float], 
                     attempt: int, max_retries: int) -> Tuple[bool, str]:
        """
        Determine if a retry is needed based on the response quality.
        
        Args:
            response: The model's response text
            extracted_num: The extracted numerical answer (None if extraction failed)
            attempt: Current attempt number (0-indexed)
            max_retries: Maximum number of retries allowed
            
        Returns:
            Tuple of (should_retry, reason) where reason is one of:
            - "incomplete_response": Response was cut off
            - "formula_only_response": Response has formulas but no final answer
            - "extraction_failed": Could not extract numerical answer
            - "": No retry needed (empty string)
        """
        if attempt >= max_retries:
            return False, ""
        
        # Check for incomplete response
        if AnswerExtractor.is_response_incomplete(response):
            return True, "incomplete_response"
        
        # Check for formula-only response
        if AnswerExtractor.is_response_formula_only(response):
            return True, "formula_only_response"
        
        # Check for extraction failure
        if extracted_num is None:
            return True, "extraction_failed"
        
        # No retry needed
        return False, ""
    
    @staticmethod
    def get_retry_temperature(base_temp: float, attempt: int, 
                             increment: float = 0.1) -> float:
        """
        Calculate temperature for retry attempt.
        
        We increase temperature slightly on retries to encourage the model
        to generate different outputs, potentially avoiding the same failure mode.
        
        Args:
            base_temp: Base temperature from configuration
            attempt: Current attempt number (0-indexed)
            increment: Temperature increase per retry
            
        Returns:
            Temperature to use for this attempt
        """
        return base_temp + (increment * attempt)
    
    @staticmethod
    def format_retry_message(attempt: int, max_retries: int, reason: str) -> str:
        """
        Format a user-friendly retry message.
        
        Args:
            attempt: Current attempt number (0-indexed)
            max_retries: Maximum retries allowed
            reason: Reason for retry
            
        Returns:
            Formatted message string
        """
        reason_messages = {
            "incomplete_response": "Response incomplete",
            "formula_only_response": "Response has formulas but no final answer",
            "extraction_failed": "Could not extract numerical answer"
        }
        
        message = reason_messages.get(reason, reason)
        return f"[Retry {attempt + 1}/{max_retries}] {message}, retrying..."