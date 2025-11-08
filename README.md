# Mathematical Reasoning with Prompt Engineering Techniques

**Author:** Shubham Bhavsar  
**Status:** 🚧 Research in Progress  
**Focus:** Evaluating and comparing prompt engineering techniques for improving mathematical reasoning in small language models

---

## 📋 Overview

This repository contains a modular research framework for systematically evaluating prompt engineering techniques on mathematical reasoning tasks using the GSM8K (Grade School Math 8K) dataset. The research focuses on understanding how different prompting strategies perform on smaller, resource-efficient language models.

### Research Questions

- How do different prompting techniques affect mathematical reasoning performance in small LLMs?
- Which strategies are most effective for resource-constrained models (1.5B-3B parameters)?
- What is the trade-off between accuracy, computational cost, and reliability?
- How do retry mechanisms and self-consistency voting impact final performance?

### Current Status

✅ **Framework Development:** Complete  
✅ **Baseline Techniques:** 4 established methods implemented  
🔬 **Novel Techniques:** Under development (paper in preparation)  
📊 **Experiments:** Large-scale evaluation in progress  
📝 **Paper:** Currently being written

---

## 🎯 Implemented Baseline Techniques

This framework currently implements and evaluates 4 well-established prompting techniques from the literature:

### 1. Zero-Shot Chain-of-Thought (Zero-Shot CoT)
**Paper:** Kojima et al., "Large Language Models are Zero-Shot Reasoners" (NeurIPS 2022)

Simple but effective approach that appends "Let's think step by step" to prompts, triggering step-by-step reasoning without requiring examples.

**Command:**
```bash
python experiments/run_zero_shot_cot.py --samples 100
```

**Citation:**
```bibtex
@inproceedings{kojima2022large,
  title={Large language models are zero-shot reasoners},
  author={Kojima, Takeshi and Gu, Shixiang Shane and Reid, Michiel and Matsuo, Yutaka and Iwasawa, Yusuke},
  booktitle={NeurIPS},
  year={2022}
}
```

---

### 2. Plan-and-Solve (PS)
**Paper:** Wang et al., "Plan-and-Solve Prompting" (ACL 2023)

Explicitly instructs the model to first devise a plan, then execute it step by step, leading to more organized reasoning.

**Command:**
```bash
python experiments/run_plan_solve.py --samples 100
```

**Citation:**
```bibtex
@inproceedings{wang2023plan,
  title={Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models},
  author={Wang, Lei and Xu, Wanyu and Lan, Yihuai and Hu, Zhiqiang and Lan, Yunshi and Lee, Roy Ka-Wei and Lim, Ee-Peng},
  booktitle={ACL},
  year={2023}
}
```

---

### 3. Plan-and-Solve Plus (PS+)
**Paper:** Wang et al., "Plan-and-Solve Prompting" (ACL 2023)

Enhanced version of Plan-and-Solve with more detailed instructions to extract variables, calculate intermediate results, and pay attention to the calculation process.

**Command:**
```bash
python experiments/run_plan_solve_plus.py --samples 100
```

---

### 4. DUP (Deeply Understanding the Problems)
**Paper:** Wu et al., arXiv 2024  
**Link:** https://arxiv.org/pdf/2404.14963

Three-step structured approach:
1. Extract the core question from the problem
2. Identify relevant problem-solving information
3. Generate solution using extracted information

**Command:**
```bash
python experiments/run_dup_prompting.py --samples 100
```

**Citation:**
```bibtex
@article{wu2024dup,
  title={DUP: A Deeper Understanding Prompting Method for Mathematical Reasoning},
  author={Wu, Yixuan and others},
  journal={arXiv preprint arXiv:2404.14963},
  year={2024}
}
```

---

## 🔬 Novel Techniques (In Development)

**⚠️ Research Under Active Development**

Additional custom prompting techniques are currently being developed and evaluated as part of ongoing research. These novel approaches show promising preliminary results and will be made public upon paper publication.

**Status:**
- ✅ Techniques designed and implemented
- 🔬 Large-scale experiments in progress
- 📊 Statistical analysis underway
- 📝 Paper in preparation
- 📅 Expected publication: TBD

**If you're interested in collaboration or early access for research purposes, please reach out.**

---

## 🏗️ Framework Architecture

### Design Philosophy

This framework is built with research reproducibility and extensibility in mind:

- **Modular Design:** Clear separation between prompts, execution strategies, and utilities
- **No Code Duplication:** 70% reduction compared to monolithic scripts
- **Easy Extension:** Add new techniques with minimal code (typically 2 files)
- **Reproducible:** Consistent evaluation pipeline across all methods
- **Well-Documented:** Academic citations and clear documentation throughout

### Directory Structure
```
gsm8k_research/
├── core/                          # Core infrastructure
│   ├── types.py                   # Data classes (GSM8KExample, ComparisonResult)
│   ├── answer_extraction.py       # Robust numerical answer extraction
│   ├── llm_manager.py             # Ollama API interface
│   └── data_loader.py             # GSM8K dataset loading & sampling
│
├── prompts/                       # Prompting strategies
│   ├── base_prompt.py             # Abstract interface
│   ├── zero_shot_cot.py           # Zero-Shot CoT implementation
│   ├── plan_solve.py              # Plan-and-Solve implementation
│   ├── plan_solve_plus.py         # Plan-and-Solve Plus implementation
│   └── dup_prompting.py           # DUP implementation
│
├── systems/                       # Execution strategies
│   ├── base_system.py             # Abstract base system
│   ├── single_step_system.py     # Single-pass reasoning
│   ├── two_step_system.py        # Two-phase reasoning
│   └── multi_step_system.py      # Multi-phase reasoning (3+ steps)
│
├── utils/                         # Reusable utilities
│   ├── retry_logic.py             # Unified retry handling
│   ├── self_consistency.py       # Majority voting implementation
│   └── result_formatter.py       # CSV output & summary statistics
│
├── experiments/                   # Experiment runners
│   ├── run_zero_shot_cot.py      # Zero-Shot CoT evaluation
│   ├── run_plan_solve.py         # Plan-and-Solve evaluation
│   ├── run_plan_solve_plus.py    # PS+ evaluation
│   └── run_dup_prompting.py      # DUP evaluation
│
└── results/                       # Organized output structure
    └── {model_name}/              # Results grouped by model
        └── {technique_name}/      # Then by technique
            └── *.csv              # Timestamped CSV files
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** 
- **[Ollama](https://ollama.ai/)** - For running local language models

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gsm8k-research.git
cd gsm8k-research

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .

# Pull a model (example - you can use any Ollama model)
ollama pull deepseek-r1:1.5b
```

### Quick Start
```bash
# Run Zero-Shot CoT on 20 samples
python experiments/run_zero_shot_cot.py --samples 20

# Run with custom configuration
python experiments/run_zero_shot_cot.py \
    --samples 100 \
    --model deepseek-r1:1.5b \
    --temp 0.0 \
    --seed 42 \
    --max-retries 2

# Disable retry mechanism
python experiments/run_plan_solve.py --samples 100 --max-retries 0

# Use self-consistency voting (5 samples, majority vote)
python experiments/run_dup_prompting.py --samples 100 --sc 5 --temp 0.7
```

---

## 📊 Dataset

### GSM8K (Grade School Math 8K)

- **Size:** 8,500+ grade school math word problems
- **Difficulty:** Problems require 2-8 reasoning steps
- **Format:** Natural language questions with detailed solutions
- **Split:** 7,473 training examples, 1,319 test examples

**Citation:**
```bibtex
@article{cobbe2021training,
  title={Training verifiers to solve math word problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

---

## 🔬 Framework Features

### 1. Unified Retry Logic

Automatically handles common failure modes:
- **Incomplete responses** - Response cut off mid-calculation
- **Formula-only responses** - Math expressions without final answer
- **Extraction failures** - Answer present but not extractable

**Control:** Use `--max-retries N` (set to 0 to disable)

**How it works:**
```python
for attempt in range(max_retries + 1):
    response = generate_response()
    number = extract_number(response)
    
    if number is not None:
        break  # Success
    
    if attempt < max_retries:
        continue  # Retry with slightly higher temperature
```

---

### 2. Self-Consistency Voting

Implements the self-consistency method (Wang et al., ICLR 2023) which samples multiple reasoning paths and selects the most consistent answer through majority voting.

**Usage:**
```bash
--sc 5 --temp 0.7  # Generate 5 samples at temperature 0.7, use majority vote
```

**Citation:**
```bibtex
@inproceedings{wang2023selfconsistency,
  title={Self-consistency improves chain of thought reasoning in language models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle={ICLR},
  year={2023}
}
```

---

### 3. Robust Answer Extraction

Multi-strategy extraction with priority order:

1. **LaTeX format:** `\boxed{42}` or `boxed{42}`
2. **Explicit markers:** "Final Answer: 42", "The answer is 42"
3. **GSM8K format:** `#### 42`
4. **Equals signs:** `= 42` (at end of line)
5. **Last number:** Final numerical value in text

**Handles:**
- Fractions: `2/3`, `1 2/3`
- Currency: `$42`, `£42`, `€42`
- Thousands: `1,234` or `1234`
- Decimals: `3.14`

---

### 4. Organized Results

Results are automatically organized in a hierarchical structure:
```
results/
├── deepseek-r1_1.5b/
│   ├── zero_shot_cot/
│   │   └── zero_shot_cot_deepseek-r1_1.5b_temp0.0_sc1_retry2_seed42_20250104_143022.csv
│   ├── plan_solve/
│   │   └── plan_solve_deepseek-r1_1.5b_temp0.0_sc1_retry2_seed42_20250104_150033.csv
│   └── dup_prompting/
│       └── dup_prompting_deepseek-r1_1.5b_temp0.0_sc1_retry2_seed42_20250104_152044.csv
│
└── llama2_7b/
    └── zero_shot_cot/
        └── zero_shot_cot_llama2_7b_temp0.0_sc1_retry2_seed42_20250104_160055.csv
```

**Benefits:**
- Easy comparison across techniques for same model
- Easy comparison across models for same technique
- Clean, navigable structure
- Timestamped files prevent overwrites

---

## 📈 Output Format

### CSV Columns

**Core Fields:**
- `original_question` - The math word problem
- `ground_truth_answer` - Full solution with step-by-step reasoning
- `ground_truth_numerical` - Correct numerical answer
- `main_response` - Model's primary response/solution
- `final_numerical` - Extracted numerical answer
- `is_correct` - Boolean indicating correctness (within 0.01 tolerance)

**Performance Metrics:**
- `processing_time` - Time per problem (seconds)
- `tokens_used` - Estimated token count (prompt + response)
- `retry_count` - Number of retries performed
- `retry_reason` - Why retry was needed (if applicable)

**Configuration:**
- `temperature` - Sampling temperature used
- `self_consistency` - Number of samples for voting
- `sc_mode` - "enabled" or "disabled"
- `sc_count` - Actual number of samples generated
- `sc_all_answers` - JSON list of all responses (for SC)
- `sc_numerical_answers` - JSON list of extracted numbers (for SC)
- `model_name` - Model identifier
- `seed` - Random seed used

**Technique-Specific Fields** (automatically filtered based on technique):
- `extraction_response` - Review/extraction step (two-step techniques)
- `plan_response` - Core question extraction (multi-step techniques)
- `info_response` - Problem information (multi-step techniques)

---

## 🎛️ Command-Line Arguments

All experiment scripts support the following arguments:

### Data Arguments
```bash
--samples N          # Number of problems to evaluate (default: 20)
--split {train,test} # Dataset split to use (default: test)
--seed N            # Random seed for reproducibility (default: 42)
```

### Model Arguments
```bash
--ollama-url URL    # Ollama server URL (default: http://localhost:11434)
--model MODEL       # Model name (default: deepseek-r1:1.5b)
--temp FLOAT        # Sampling temperature (default: 0.0)
```

### Technique Arguments
```bash
--sc N              # Self-consistency count (default: 1, i.e., disabled)
--max-retries N     # Maximum retry attempts (default: 2, set 0 to disable)
```

### Output Arguments
```bash
--output-dir DIR    # Results directory (default: results/)
```

### Example Configurations
```bash
# Greedy decoding, no retry, 100 samples
python experiments/run_zero_shot_cot.py \
    --samples 100 --temp 0.0 --max-retries 0

# Self-consistency with 5 samples
python experiments/run_plan_solve.py \
    --samples 100 --sc 5 --temp 0.7

# Full test set evaluation
python experiments/run_dup_prompting.py \
    --samples 1319 --split test --seed 42

# Different model
python experiments/run_zero_shot_cot.py \
    --model llama2:7b --samples 50
```

---

## 🔧 Extending the Framework

The modular design makes it easy to add new prompting techniques:

### For Single-Step Techniques

**1. Create prompt class:**
```python
# prompts/my_technique.py
from prompts.base_prompt import BasePrompt

class MyTechniquePrompt(BasePrompt):
    @property
    def name(self) -> str:
        return "my_technique"
    
    @property
    def display_name(self) -> str:
        return "My Technique Name"
    
    @property
    def citation(self) -> str:
        return """@inproceedings{...}"""
    
    @property
    def description(self) -> str:
        return "Brief description of the technique"
    
    def format_prompt(self, question: str, **kwargs) -> str:
        return f"Custom prompt format: {question}"
```

**2. Create experiment runner:**
```python
# experiments/run_my_technique.py
from prompts.my_technique import MyTechniquePrompt
from systems.single_step_system import SingleStepSystem

# Use existing SingleStepSystem
# Copy structure from run_zero_shot_cot.py
```

**3. Run:**
```bash
python experiments/run_my_technique.py --samples 100
```

### For Multi-Step Techniques

For techniques requiring multiple LLM calls (like DUP's 3-step approach):

1. Implement `format_plan_prompt()`, `format_info_prompt()`, `format_solve_prompt()` methods
2. Use `MultiStepSystem` instead of `SingleStepSystem`
3. See `prompts/dup_prompting.py` for reference

---

## 📚 Research Methodology

### Evaluation Protocol

1. **Dataset Sampling:** Random sampling with fixed seed for reproducibility
2. **Answer Extraction:** Unified extraction logic across all techniques
3. **Correctness:** Numerical comparison with 0.01 tolerance
4. **Metrics:** Accuracy, processing time, token usage, retry statistics

### Experimental Controls

- **Same dataset samples** across all techniques (when using same seed)
- **Same model** for fair comparison
- **Same extraction logic** to isolate prompting effects
- **Configurable retry and self-consistency** for ablation studies

### Reproducibility

All experiments are fully reproducible:
- Fixed random seeds
- Deterministic sampling (when temp=0.0)
- Logged configurations in filenames
- Complete results saved to CSV

---

## 📊 Current Research Focus

### Active Experiments

- Comprehensive evaluation across multiple models
- Ablation studies on retry mechanisms
- Self-consistency impact analysis
- Computational efficiency comparisons

### Models Under Evaluation

- DeepSeek-R1 (1.5B, 7B)
- Llama family (various sizes)
- Other open-source models via Ollama

### Upcoming Analysis

- Statistical significance testing
- Error analysis by problem type
- Computational cost vs. accuracy trade-offs
- Failure mode categorization

---

## 📝 Publication Status

**Paper in preparation** - Expected submission in 2025

This research will include:
- Comprehensive comparison of baseline techniques
- Novel prompting approaches (currently under development)
- Large-scale empirical results
- Analysis of small language model capabilities
- Practical insights for resource-constrained deployment

**Pre-print and full code release planned upon paper submission.**

---

## 🤝 Collaboration & Contact

**Interested in this research?**

I'm open to:
- Research collaborations
- Discussion of preliminary results
- Sharing insights on prompt engineering
- Joint experiments on related problems

**Contact:**
- **GitHub:** [@shubhambhavsar](https://github.com/shubhambhavsar)
- **LinkedIn:** [Shubham Bhavsar](https://www.linkedin.com/in/shubham-bhavsar/)
- **Email:** sbhavsar3798@gmail.com

---

## 🙏 Acknowledgments

- **GSM8K Dataset:** OpenAI for the comprehensive benchmark
- **Ollama:** For making local LLM experimentation accessible
- **Original Authors:** All researchers whose techniques are implemented here

---

## 📜 License

**Code:** MIT License - Feel free to use the framework for your research

**Novel Techniques:** Research in progress - please contact for collaboration

---

## ⭐ Star This Repo

If you find this research interesting or useful for your work, please star the repository to stay updated on:
- New technique implementations
- Experimental results
- Paper publication
- Framework improvements

---

**Last Updated:** January 2025  
**Status:** 🚧 Active Development - Check back for updates!