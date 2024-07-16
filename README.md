# ECCO

This repository contains the source code for the paper "ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness?"


<div align="center">
  <img src="https://github.com/user-attachments/assets/7414638d-2aa2-4597-97c9-6711c7dff64a" alt="ECCO Image">
</div>

## Dataset
The dataset is available on Huggingface at: [EfficientCode/ECCO](https://huggingface.co/datasets/EfficientCode/ECCO).

It consists of 2 subsets `edit` and `generate` each with 3 splits (`train`, `val` and `test`).

### Loading the dataset 
```python
dataset = load_dataset('EfficientCode/ECCO', 'edit') # For history-based editing setting
dataset = load_dataset('EfficientCode/ECCO', 'generate') # For nl-instructed generation setting
```

## Experiments

### Environment setup
```bash 
conda env create -f environment.yml
conda activate ecco
```

### Code structure 
The `src/` folder consists of the primary codebase:
1. `src/evaluation` consists of scripts to run evaluation of model generated code on the Judge0 environment server hosted on AWS. Please see instructions to setup the evaluation server.
   - `edit_eval.py` is the script for evaluating code generated on the metrics for the history-based editing setting
   - `generate_eval.py` is the script for evaluating code generated on the metrics for the NL-instructed generation setting
2. `src/experiments` consists of the scripts to run modelling experiment. 
   - `model_classes.py` consists of the Inference Engine Classes for each model that is benchmarked.
   - `inference.py` is the entrypoint for running the experiments
   - `prompt_formats.py` and `utils.py` cotains utilities for prompt building and execution feedback formatting

### Setting up the Judge0 evaluation setup 
Amazon Machine Image (AMI) and instructions coming soon!

<div align="center">
  <img src="https://github.com/user-attachments/assets/930ee4f3-c120-4ad6-b4df-35c3e2c3fbca" alt="Judge0 Setup Image">
  </div>
