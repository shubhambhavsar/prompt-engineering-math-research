"""
Data types for GSM8K evaluation experiments.

This module contains the core data structures used across all prompting techniques.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GSM8KExample:
    """A single GSM8K problem instance."""
    question: str
    answer: str  # Ground truth answer with reasoning
    numerical_answer: float  # Extracted numerical value


@dataclass
class ComparisonResult:
    """Results from evaluating a single GSM8K example."""
    
    # Input
    original_question: str
    ground_truth_answer: str
    ground_truth_numerical: float
    
    # Model outputs
    main_response: str  # Primary response text (or solution for multi-step)
    final_numerical: Optional[float]  # Extracted numerical answer
    
    # Evaluation
    is_correct: bool
    processing_time: float  # Seconds
    tokens_used: int
    
    # Configuration
    temperature: float
    self_consistency: int
    sc_mode: str  # "enabled" or "disabled"
    
    # Self-consistency details (if applicable)
    sc_count: int  # Number of samples generated
    sc_all_answers: str  # JSON list of all response texts
    sc_numerical_answers: str  # JSON list of all extracted numbers
    
    # Model info
    model_name: str
    
    # Retry info
    retry_count: int = 0
    retry_reason: str = ""
    
    # Additional response fields (for multi-step approaches)
    extraction_response: str = ""  # For two-step methods
    plan_response: str = ""  # For DUP-style methods
    info_response: str = ""  # For DUP-style methods