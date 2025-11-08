"""
Prompting strategies for mathematical reasoning.

This package contains implementations of various prompting techniques
for improving mathematical reasoning in language models.
"""

from .base_prompt import BasePrompt
from .zero_shot_cot import ZeroShotCoTPrompt

__all__ = [
    'BasePrompt',
    'ZeroShotCoTPrompt',
    'PlanSolvePrompt',
    'PlanSolvePlusPrompt',
    'TwoStepReviewPrompt',
    'DUPPrompt',
]