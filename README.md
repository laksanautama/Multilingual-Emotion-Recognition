# Balinese Language Emotion Recognition -- (BaLER)
This repository contains experiments, code, and resources for emotion recognition in the Balinese language.
The project investigates several modelling strategies, ranging from multilingual language models (LMs) to Large Language Models (LLMs) with different adaptation methods (Few-Shot, RAG, LoRA).
The main goal is to explore how modern representation and generative models perform on low-resource languages such as Balinese.

## Task
We propose 2 methods for this task: 
- Crosslingual Emotion Recognition: We trained languages that belong to the same family with Balinese such as Indonesian (ind), Javanese (jav), and Sundanese (sun).
- LLM-based Emotion Recognition: We used several adaptation methods: `Few-shot`, `RAG`, and `Lora`.

## Dataset
Our Balinese-text emotion dataset was created by extracting text from Balinese storiette and then manually labeling the emotions. For the related family languages (Indonesian, Javanese, and Sundanese), the emotion data was sourced from the publicly available Brighter Dataset [see paper](https://arxiv.org/abs/2502.11926)

## Features
- The language models that have been tested so far:
  - indolem/indobert-base-uncased (default)
  - XLM-Large
  - LaBSE
- Works on these LLMs:
  - gemini-2.5-flash (default)
  - DeepSeek-Reasoner
  - Qwen3
- Multiple adaptation/fine-tuning methods:
  - Few-shot (default)
  - RAG
  - Lora

## Requirements
<pre>pip install -r requirements.txt</pre>
For LLM inference (Gemini, etc.) and also for loading Brighter Dataset from Huggingface hub, set your API keys accordingly.
This project uses several external APIs for model loading and LLM inference.
Create a `.env` file in the project root to store your keys securely, and add your API keys to `.env` file:
<pre>
  # Required: HuggingFace access token
HUGGINGFACE_TOKEN=your_hf_token_here

# Required if using Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Groq API (for Qwen/Qwen3 or Mixtral models)
GROQ_API_KEY=your_groq_api_key_here

# Optional: DeepSeek API (for DeepSeek-V3)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
</pre>

## How to Run the Project
All experiments are executed using the `main.py` file.
The main arguments are:
| Arguments | Description | Default |
|-----------|-------------|---------|
| --lm     | Preferred language model for embedding/classification | `'indolem/indobert-base-uncased'` |
| --llm    | Large language model used for generation/analysis | `'gemini-2.5-flash'`|
| --adaptation_method | Method for fine-tuned the LLM | `'few_shot'`|

### ✅ Basic Command
<pre>python main.py</pre>
This will run the experiment using all default settings:
- LM: `indolem/indobert-base-uncased`
- LLM: `gemini-2.5-flash`
- Adaptation: `few_shot`

### ✅ Example using other LM, LLM, and Adaptation Method
<pre>python main.py --lm "sentence-transformers/LaBSE" --llm "qwen/qwen3-32b" --adaptation_method "rag"</pre>
This will run the experiment using:
- LM : `LaBSE'
- LLM : `qwen/qwen3-32b`
- Adaptation : `rag`

## 📜 Citation
@misc{muhammad2025brighterbridginggaphumanannotated,
      title={BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages}, 
      author={Shamsuddeen Hassan Muhammad and Nedjma Ousidhoum and Idris Abdulmumin and Jan Philip Wahle and Terry Ruas and Meriem Beloucif and Christine de Kock and Nirmal Surange and Daniela Teodorescu and Ibrahim Said Ahmad and David Ifeoluwa Adelani and Alham Fikri Aji and Felermino D. M. A. Ali and Ilseyar Alimova and Vladimir Araujo and Nikolay Babakov and Naomi Baes and Ana-Maria Bucur and Andiswa Bukula and Guanqun Cao and Rodrigo Tufino Cardenas and Rendi Chevi and Chiamaka Ijeoma Chukwuneke and Alexandra Ciobotaru and Daryna Dementieva and Murja Sani Gadanya and Robert Geislinger and Bela Gipp and Oumaima Hourrane and Oana Ignat and Falalu Ibrahim Lawan and Rooweither Mabuya and Rahmad Mahendra and Vukosi Marivate and Alexander Panchenko and Andrew Piper and Charles Henrique Porto Ferreira and Vitaly Protasov and Samuel Rutunda and Manish Shrivastava and Aura Cristina Udrea and Lilian Diana Awuor Wanzare and Sophie Wu and Florian Valentin Wunderlich and Hanif Muhammad Zhafran and Tianhui Zhang and Yi Zhou and Saif M. Mohammad},
      year={2025},
      eprint={2502.11926},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11926}, 
}
 




