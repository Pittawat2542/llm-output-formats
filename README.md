# BenchING: A Benchmark for Evaluating Large Language Models in Following Structured Output Format Instruction in Text-Based Narrative Game Tasks

This repository contains the code and datasets for the paper "BenchING: A Benchmark for Evaluating Large Language Models in Following Structured Output Format Instruction in Text-Based Narrative Game Tasks" accepted for [IEEE ToG](https://transactions.games).

## Authors
Pittawat Taveekitworachai, Mury F. Dewantoro, Yi Xia, Pratch Suntichaikul, and Ruck Thawonmas

## Abstract

This paper presents BenchING, a new benchmark for evaluating large language models (LLMs) on their ability to follow structured output format instructions in text-based procedural content generation (PCG) tasks. The ability to condition LLMs to output in specified formats proves useful, as downstream components in LLM-integrated games often require structured outputs for exchanging information. However, there is a gap in evaluating this aspect of LLMs, especially in narrative PCG tasks, making it difficult to select LLMs and design games or applications integrating these LLMs. To demonstrate the potential of our benchmark, we evaluate nine LLMs for their ability to generate parseable formatted outputs using five selected text-based PCG tasks. We report on the performance of these LLMs on these tasks. Additionally, we categorize more detailed error types and propose solutions by utilizing LLMs to fix these errors. We also conduct a scaling study, investigating an emergent point of LLMs for their ability to fix malformed formatted content using eight quantized LLMs with varying original sizes from 0.62B to 72.3B. Furthermore, we perform a qualitative study to assess the quality of the generated content. We make our source code and raw data available for future research.

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n benching python=3.12
```
and activate it:
```bash
conda activate benching
```
1. Copy `.env.example` and rename it to `.env`. Follow instructions on [this page](https://platform.openai.com/docs/api-reference/authentication) to obtain your own OpenAI API key. Add API keys of other LLMs as needed.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Run the code.
```bash
python main.py [command]
```
