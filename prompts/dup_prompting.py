"""
DUP (Deeply Understanding the Problems) Prompting

Paper: "DUP: A Deeper Understanding Prompting Method for Mathematical Reasoning"
Authors: Wu et al. (2024)
Venue: arXiv preprint arXiv:2404.14963

LaTeX Citation:
@article{wu2024dup,
  title={DUP: A Deeper Understanding Prompting Method for Mathematical Reasoning},
  author={Wu, Yixuan and others},
  journal={arXiv preprint arXiv:2404.14963},
  year={2024}
}

Technique Description:
DUP (Deeply Understanding the Problems) prompting uses a three-step approach to 
enhance mathematical reasoning:
1. Extract the core question from the problem
2. Extract relevant problem-solving information
3. Generate solution based on core question and extracted information

This structured decomposition helps the model better understand the problem before 
solving it, leading to improved accuracy on mathematical reasoning tasks.
"""

from prompts.base_prompt import BasePrompt


class DUPPrompt(BasePrompt):
    """DUP (Deeply Understanding the Problems) prompting implementation."""
    
    @property
    def name(self) -> str:
        return "dup_prompting"
    
    @property
    def display_name(self) -> str:
        return "DUP (Deeply Understanding the Problems)"
    
    @property
    def citation(self) -> str:
        return """@article{wu2024dup,
  title={DUP: A Deeper Understanding Prompting Method for Mathematical Reasoning},
  author={Wu, Yixuan and others},
  journal={arXiv preprint arXiv:2404.14963},
  year={2024}
}"""
    
    @property
    def description(self) -> str:
        return (
            "DUP (Deeply Understanding the Problems) uses a three-step approach: "
            "1) Extract the core question, 2) Extract problem-solving information, "
            "3) Generate solution based on extracted information. This structured "
            "decomposition helps the model better understand problems before solving."
        )
    
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        This method is not used for three-step prompts.
        Use format_plan_prompt, format_info_prompt, and format_solve_prompt instead.
        """
        return self.format_plan_prompt(question)
    
    def format_plan_prompt(self, question: str) -> str:
        """
        Format Step 1: Extract the core question.
        
        Args:
            question: The math word problem
            
        Returns:
            Formatted prompt for extracting core question
        """
        return f"""Problem: {question}

Please extract the core question from the problem above. Only extract the most comprehensive and detailed core question that needs to be answered.

Core Question:"""
    
    def format_info_prompt(self, question: str, core_question: str) -> str:
        """
        Format Step 2: Extract problem-solving information.
        
        Args:
            question: The original math word problem
            core_question: The extracted core question from Step 1
            
        Returns:
            Formatted prompt for extracting problem-solving information
        """
        return f"""Problem: {question}

Core Question: {core_question}

Note: Please extract the problem-solving information related to the core question above. Only extract the most useful information. List them one by one!

Problem-solving Information:"""
    
    def format_solve_prompt(self, question: str, core_question: str, problem_info: str) -> str:
        """
        Format Step 3: Generate solution using extracted information.
        
        Args:
            question: The original math word problem
            core_question: The extracted core question from Step 1
            problem_info: The extracted problem-solving information from Step 2
            
        Returns:
            Formatted prompt for generating solution
        """
        return f"""Problem: {question}

Core Question: {core_question}

Problem-solving Information:
{problem_info}

Based on the core question and problem-solving information above, solve the problem step by step and provide the final numerical answer. Make sure to clearly state the final answer.

Solution:"""