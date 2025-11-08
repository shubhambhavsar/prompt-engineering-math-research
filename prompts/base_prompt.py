"""
Base abstract class for prompting strategies.

This module defines the interface that all prompting techniques must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePrompt(ABC):
    """
    Abstract base class for all prompting strategies.
    
    Each prompting technique (Zero-Shot CoT, Plan-and-Solve, etc.) implements
    this interface to provide a consistent API for the reasoning systems.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the prompting technique.
        
        This should be a short, URL-safe identifier (e.g., "zero_shot_cot").
        Used for logging and file naming.
        """
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Return the full display name of the prompting technique.
        
        This should be the formal name as it appears in papers
        (e.g., "Zero-Shot Chain-of-Thought").
        """
        pass
    
    @property
    @abstractmethod
    def citation(self) -> str:
        """
        Return the LaTeX citation for the technique's paper.
        
        Returns a BibTeX-formatted citation string.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a brief description of the technique.
        
        Should be 2-4 sentences explaining the core idea.
        """
        pass
    
    @abstractmethod
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        Format a question into a prompt using this technique.
        
        Args:
            question: The math word problem to solve
            **kwargs: Additional parameters specific to the technique
            
        Returns:
            Formatted prompt string ready for the LLM
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this prompting technique.
        
        Returns:
            Dictionary with name, display_name, citation, and description
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "citation": self.citation,
            "description": self.description
        }