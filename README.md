# 🧠 Roberta-based Text Classification with LoRA
# Deep Learning CS 6953 / ECE 7123 Spring 2025 (Project 2)

This repository contains a notebook to fine-tune a RoBERTa model using parameter-efficient fine-tuning (LoRA) on a text classification task. The notebook demonstrates setup, model loading, data processing, training, and evaluation.
Also, the repository includes a project report detailing the methodology, experiments, results, and key insights from the fine-tuning process.

Competition Link: https://www.kaggle.com/competitions/deep-learning-spring-2025-project-2/overview
## Team Members
- Samridh Srivastava (ss18906)
- Krittin Nagar (kn2670)
- Shikhar Malik (sm12762)

## 📦 Installation

Install the required libraries before running the notebook:

```bash
pip install transformers datasets evaluate accelerate peft trl bitsandbytes
pip install nvidia-ml-py3
```

## 🧪 Libraries Used

```python
import os
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset, ClassLabel
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

## 📁 Project Structure

```
notebook.ipynb      # Jupyter notebook with the entire workflow
README.md           # This file
projectReport.pdf   # Project Report
```
