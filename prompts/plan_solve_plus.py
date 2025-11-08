"""
Plan-and-Solve Plus Prompting

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
Plan-and-Solve Plus (PS+) extends the basic Plan-and-Solve prompting with more 
detailed instructions. It explicitly guides the model to extract relevant variables 
and their values, calculate intermediate results, and pay attention to the 
calculation process. These additional instructions help reduce calculation errors 
and missing reasoning steps, leading to further improvements in mathematical 
reasoning accuracy.
"""

from prompts.base_prompt import BasePrompt


class PlanSolvePlusPrompt(BasePrompt):
    """Plan-and-Solve Plus prompting with enhanced instructions."""
    
    @property
    def name(self) -> str:
        return "plan_solve_plus"
    
    @property
    def display_name(self) -> str:
        return "Plan-and-Solve Plus"
    
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
            "Plan-and-Solve Plus extends basic Plan-and-Solve with more detailed "
            "instructions. It explicitly guides the model to extract relevant variables, "
            "calculate intermediate results, and pay attention to calculations, reducing "
            "errors and missing reasoning steps."
        )
    
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        Format question with Plan-and-Solve Plus prompt.
        
        Args:
            question: The math word problem
            
        Returns:
            Formatted prompt with detailed planning instructions
        """
        return f"""Let's solve this math problem step by step.

Question: {question}

First, devise a plan to solve the problem. Then carry out the plan to solve the problem step by step. Pay attention to common sense and arithmetic. When solving the problem:
1. Extract relevant variables and their corresponding numerals
2. Calculate intermediate results step by step
3. Pay attention to the calculation process

Show your work and provide the final answer."""