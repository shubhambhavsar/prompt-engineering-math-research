"""
Plan-and-Solve Prompting

Paper: "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning 
        by Large Language Models"
Authors: Wang et al. (2023)
Venue: ACL 2023

LaTeX Citation:
@inproceedings{wang2023plan,
  title={Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models},
  author={Wang, Lei and Xu, Wanyu and Lan, Yihuai and Hu, Zhiqiang and Lan, Yunshi and Lee, Roy Ka-Wei and Lim, Ee-Peng},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  pages={2609--2634},
  year={2023}
}

Technique Description:
Plan-and-Solve prompting extends Zero-Shot CoT by explicitly instructing the model 
to first devise a plan and then carry out that plan step by step. This two-phase 
instruction helps the model organize its reasoning more systematically, leading to 
improved accuracy on mathematical reasoning tasks compared to simple "let's think 
step by step" prompting.
"""

from prompts.base_prompt import BasePrompt


class PlanSolvePrompt(BasePrompt):
    """Basic Plan-and-Solve prompting implementation."""
    
    @property
    def name(self) -> str:
        return "plan_solve"
    
    @property
    def display_name(self) -> str:
        return "Plan-and-Solve"
    
    @property
    def citation(self) -> str:
        return """@inproceedings{wang2023plan,
  title={Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models},
  author={Wang, Lei and Xu, Wanyu and Lan, Yihuai and Hu, Zhiqiang and Lan, Yunshi and Lee, Roy Ka-Wei and Lim, Ee-Peng},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  pages={2609--2634},
  year={2023}
}"""
    
    @property
    def description(self) -> str:
        return (
            "Plan-and-Solve prompting extends Zero-Shot CoT by explicitly instructing "
            "the model to first devise a plan and then carry out that plan step by step. "
            "This two-phase instruction helps organize reasoning more systematically."
        )
    
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        Format question with Plan-and-Solve prompt.
        
        Args:
            question: The math word problem
            
        Returns:
            Formatted prompt with planning instructions
        """
        return f"""Let's solve this math problem step by step.

Question: {question}

First, devise a plan to solve the problem. Then carry out the plan to solve the problem step by step.

Show your work and provide the final answer."""