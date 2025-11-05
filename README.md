# Balinese Language Emotion Recognition -- (BaLER)
This repository contains experiments, code, and resources for emotion recognition in the Balinese language.
The project investigates several modelling strategies, ranging from multilingual language models (LMs) to Large Language Models (LLMs) with different adaptation methods (Few-Shot, RAG, LoRA).
The main goal is to explore how modern representation and generative models perform on low-resource languages such as Balinese.

## Task
We propose 2 methods for this task: 
- Crosslingual Emotion Recognition: Train languages that belong to the same family with Balinese such as Indonesian (ind), Javanese (jav), and Sundanese (sun).
- LLM-based Emotion Recognition: We use several adaptation methods: Few-shot, RAG, and Lora.

## Dataset
Our Balinese-text emotion dataset was created by extracting text from Balinese stories and then manually labeling the emotions. For the related family languages (Indonesian, Javanese, and Sundanese), the emotion data was sourced from the publicly available Brighter Dataset (https://arxiv.org/abs/2502.11926) 

## Features
- The language models that have been tested so far:
  - indolem/indobert-base-uncased (default)
  - XLM-Large
  - LaBSE
 




