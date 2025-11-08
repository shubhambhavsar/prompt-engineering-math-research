"""
GSM8K dataset loader.

This module handles loading and sampling from the GSM8K dataset using
the Hugging Face datasets library.
"""

import random
import re
from typing import List
from datasets import load_dataset

from .types import GSM8KExample


class GSM8KDataLoader:
    """
    Loader for GSM8K (Grade School Math 8K) dataset.
    
    GSM8K is a dataset of 8.5K high quality grade school math word problems.
    Each problem requires 2-8 steps to solve and includes a natural language
    solution with the final numerical answer.
    
    Reference:
        Cobbe et al. "Training Verifiers to Solve Math Word Problems"
        arXiv:2110.14168 (2021)
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.dataset = None

    def load_dataset_split(self, split: str = "test"):
        """
        Load a specific split of the GSM8K dataset.
        
        Args:
            split: Dataset split to load ("train" or "test")
            
        Returns:
            The loaded dataset split
        """
        if self.dataset is None:
            self.dataset = load_dataset("openai/gsm8k", "main", split=split)
        return self.dataset

    def load_sample_data(self, n_samples: int = 50, split: str = "test", 
                        seed: int = 42) -> List[GSM8KExample]:
        """
        Load a random sample from the GSM8K dataset.
        
        Args:
            n_samples: Number of samples to load
            split: Dataset split ("train" or "test")
            seed: Random seed for reproducibility
            
        Returns:
            List of GSM8KExample instances
        """
        ds = self.load_dataset_split(split)
        
        # Cap at dataset size
        if n_samples > len(ds):
            n_samples = len(ds)
        
        # Sample random indices
        random.seed(seed)
        indices = random.sample(range(len(ds)), n_samples)
        
        examples = []
        for idx in indices:
            item = ds[idx]
            question = item['question']
            answer = item['answer']
            
            # Extract numerical answer from "#### NUMBER" format
            match = re.search(r'####\s*([\d,\.]+)', answer)
            if match:
                try:
                    numerical_answer = float(match.group(1).replace(',', ''))
                    examples.append(GSM8KExample(
                        question=question,
                        answer=answer,
                        numerical_answer=numerical_answer
                    ))
                except ValueError:
                    # Skip examples where numerical extraction fails
                    continue
        
        return examples

    def load_full_split(self, split: str = "test") -> List[GSM8KExample]:
        """
        Load the entire split of the GSM8K dataset.
        
        Args:
            split: Dataset split ("train" or "test")
            
        Returns:
            List of all GSM8KExample instances in the split
        """
        ds = self.load_dataset_split(split)
        
        examples = []
        for item in ds:
            question = item['question']
            answer = item['answer']
            
            match = re.search(r'####\s*([\d,\.]+)', answer)
            if match:
                try:
                    numerical_answer = float(match.group(1).replace(',', ''))
                    examples.append(GSM8KExample(
                        question=question,
                        answer=answer,
                        numerical_answer=numerical_answer
                    ))
                except ValueError:
                    continue
        
        return examples