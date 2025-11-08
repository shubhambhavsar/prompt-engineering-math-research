"""
Zero-Shot Chain-of-Thought Prompting

Paper: "Large Language Models are Zero-Shot Reasoners"
Authors: Kojima et al. (2022)
Venue: NeurIPS 2022

LaTeX Citation:
@inproceedings{kojima2022large,
  title={Large language models are zero-shot reasoners},
  author={Kojima, Takeshi and Gu, Shixiang Shane and Reid, Michiel and 
          Matsuo, Yutaka and Iwasawa, Yusuke},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={22199--22213},
  year={2022}
}

Technique Description:
Zero-Shot CoT appends "Let's think step by step" to the question, prompting 
the model to generate intermediate reasoning steps before arriving at the 
final answer. This simple modification significantly improves reasoning 
performance without requiring example demonstrations or few-shot learning.

The technique works by triggering the model's step-by-step reasoning 
capabilities through a simple prompt suffix, making it broadly applicable 
across various reasoning tasks without task-specific engineering.
"""

from prompts.base_prompt import BasePrompt


class ZeroShotCoTPrompt(BasePrompt):
    """Zero-Shot Chain-of-Thought prompting implementation."""
    
    @property
    def name(self) -> str:
        return "zero_shot_cot"
    
    @property
    def display_name(self) -> str:
        return "Zero-Shot Chain-of-Thought"
    
    @property
    def citation(self) -> str:
        return """@inproceedings{kojima2022large,
  title={Large language models are zero-shot reasoners},
  author={Kojima, Takeshi and Gu, Shixiang Shane and Reid, Michiel and 
          Matsuo, Yutaka and Iwasawa, Yusuke},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={22199--22213},
  year={2022}
}"""
    
    @property
    def description(self) -> str:
        return (
            "Zero-Shot CoT appends 'Let's think step by step' to the question, "
            "prompting the model to generate intermediate reasoning steps before "
            "arriving at the final answer. This simple modification significantly "
            "improves reasoning performance without requiring example demonstrations."
        )
    
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        Format question with Zero-Shot CoT trigger.
        
        Args:
            question: The math word problem
            
        Returns:
            Formatted prompt: "Q: {question}\\n\\nA: Let's think step by step.\\n"
        """
        return f"Q: {question}\n\nA: Let's think step by step.\n"