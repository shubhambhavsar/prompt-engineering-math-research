"""
Self-consistency voting mechanism for mathematical reasoning.

This module implements majority voting over multiple sampled responses
to improve answer accuracy through ensemble methods.

Reference:
    Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in 
    Language Models" ICLR 2023
"""

from typing import List, Optional, Tuple
from collections import Counter


class SelfConsistency:
    """
    Self-consistency voting for selecting the most reliable answer.
    
    Self-consistency samples multiple reasoning paths at higher temperature
    and selects the most consistent answer through majority voting. This
    improves accuracy by marginalizing over diverse reasoning paths.
    """
    
    @staticmethod
    def majority_vote(numerical_answers: List[Optional[float]], 
                     all_responses: List[str],
                     tolerance: float = 0.01) -> Tuple[int, float, int]:
        """
        Perform majority voting on numerical answers.
        
        Args:
            numerical_answers: List of extracted numerical answers (may contain None)
            all_responses: Corresponding list of response texts
            tolerance: Tolerance for considering two numbers equal (default 0.01)
            
        Returns:
            Tuple of (winning_index, winning_value, vote_count)
            - winning_index: Index of the response with the majority answer
            - winning_value: The numerical value that won
            - vote_count: Number of votes for the winning answer
        """
        if not numerical_answers:
            return 0, 0.0, 0
        
        # Round to handle floating point precision issues
        normalized_counts = Counter()
        normalized_map = {}  # Maps rounded value to list of original indices
        
        for i, num in enumerate(numerical_answers):
            if num is not None:
                # Round to nearest tolerance
                normalized_val = round(num / tolerance) * tolerance
                normalized_counts[normalized_val] += 1
                normalized_map.setdefault(normalized_val, []).append(i)
        
        # If no valid numbers extracted, return first response
        if not normalized_counts:
            return 0, 0.0, 0
        
        # Get the most common answer
        most_common_val, vote_count = normalized_counts.most_common(1)[0]
        
        # Get indices that voted for this answer
        majority_indices = normalized_map[most_common_val]
        
        # Return the first index with the winning answer
        winning_index = majority_indices[0]
        
        return winning_index, most_common_val, vote_count
    
    @staticmethod
    def calculate_agreement_rate(numerical_answers: List[Optional[float]],
                                tolerance: float = 0.01) -> float:
        """
        Calculate the agreement rate among sampled answers.
        
        Args:
            numerical_answers: List of extracted numerical answers
            tolerance: Tolerance for considering two numbers equal
            
        Returns:
            Agreement rate as a fraction (0.0 to 1.0)
        """
        valid_answers = [num for num in numerical_answers if num is not None]
        
        if len(valid_answers) < 2:
            return 1.0 if valid_answers else 0.0
        
        # Count agreements
        normalized_counts = Counter()
        for num in valid_answers:
            normalized_val = round(num / tolerance) * tolerance
            normalized_counts[normalized_val] += 1
        
        # Agreement rate is the fraction of answers that match the majority
        most_common_count = normalized_counts.most_common(1)[0][1]
        return most_common_count / len(valid_answers)