import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from .llm_configs import PROMPT_CONFIG, TRANSLATION_MAP
from .model_config import DATA_CONFIG
import json
import re
from collections import Counter


def select_language_config(prompt_language: str):
    """Select the appropriate prompt configuration based on the specified language."""

    if prompt_language.lower() == 'indonesian' or prompt_language.lower() == 'ind':
        
        config = {
                    'example_template' : PROMPT_CONFIG["IND_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["IND_PREFIX"],
                    'suffix' : PROMPT_CONFIG["IND_SUFFIX"],
                    'clf_template': PROMPT_CONFIG["IND_CLF_TEMPLATE"],
                    'zero_shot_prefix': PROMPT_CONFIG["IND_ZERO_SHOT_PREFIX"],
                    'label_column' : DATA_CONFIG["IND_LABELS"],
                    'input' : "teks_masukan",
                    'answer' : "jawaban",
                    'conclusion' : "kesimpulan",
                    'reason' : "alasan"

                    }
        
    elif prompt_language.lower() == 'balinese' or prompt_language.lower() == 'bal':

        config = {
                    'example_template' : PROMPT_CONFIG["BAL_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["BAL_PREFIX"],
                    'suffix' : PROMPT_CONFIG["BAL_SUFFIX"],
                    'clf_template' : PROMPT_CONFIG["BAL_CLF_TEMPLATE"],
                    'zero_shot_prefix': PROMPT_CONFIG["BAL_ZERO_SHOT_PREFIX"],
                    'label_column' : DATA_CONFIG["BAL_LABELS"],
                    'input' : "masukan",
                    'answer' : "pasaut",
                    'conclusion' : "simpulan",
                    'reason' : "alasan"

                    }

    else:  # Default to English
        
        config = {
                    'example_template' : PROMPT_CONFIG["EN_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["EN_PREFIX"],
                    'suffix' : PROMPT_CONFIG["EN_SUFFIX"],
                    'clf_template' : PROMPT_CONFIG["BIN_CLF_TEMPLATE"],
                    'zero_shot_prefix': PROMPT_CONFIG["EN_ZERO_SHOT_PREFIX"],
                    'label_column' : DATA_CONFIG["LABELS"],
                    'input' : "input",
                    'answer' : "answer",
                    'conclusion' : 'conclusion',
                    'reason' : "reason"

                    }

    return config


def convert_output_text(output_text: str, prompt_language: str):
    """Convert the output text from the LLM to standardized 'yes' or 'no'."""
    output_text = output_text.strip().lower()
    if prompt_language.lower() == 'indonesian' or prompt_language.lower() == 'ind':
        if 'jawaban: ya' in output_text:
            return 'yes'
        elif 'jawaban: tidak' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'ya' nor 'tidak' is found
            return 'no'
    elif prompt_language.lower() == 'balinese' or prompt_language.lower() == 'bal':
        if 'pasaut: inggih' in output_text:
            return 'yes'
        elif 'pasaut: ten' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'inggih' nor 'ten' is found
            return 'no'
        
    else:  # Default to English
        if 'answer: yes' in output_text:
            return 'yes'
        elif 'answer: no' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'yes' nor 'no' is found
            return 'no'

def classify_text(text, chain, prompt_language: str, emotion: str = None):
    """Classify the input text using the provided LLM chain."""
    if emotion:
        result = chain.invoke({"input": text, "emotion": emotion})
    else:
        result = chain.invoke({"input": text})
    
    output_text = result.content.strip().lower()

    # print(f"{output_text}")

    return convert_output_text(output_text, prompt_language), output_text

def extract_revision(text):
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)

    answer = answer_match.group(1) if answer_match else ""
    reason = reason_match.group(1) if reason_match else ""

    return answer, reason


def extract_feedback_confidence(text):
    # Capture everything between "feedback":
    feedback_pattern = r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"'
    confidence_pattern = r'"confidence"\s*:\s*"((?:[^"\\]|\\.)*)"'

    feedback_match = re.search(feedback_pattern, text, flags=re.DOTALL)
    confidence_match = re.search(confidence_pattern, text)

    feedback = feedback_match.group(1) if feedback_match else ""
    confidence = confidence_match.group(1) if confidence_match else ""

    feedback = feedback.encode('utf-8').decode('unicode_escape')

    return feedback, confidence

def getPeersReview(model, answer:str, reason:str, text:str, emotion:str, examples):
    
    ex_prompt = PromptTemplate(
                input_variables=['input', 'answer', 'conclusion'],
                template=PROMPT_CONFIG["EN_EXAMPLE_TEMPLATE"]
                )
    review_prompt = FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=ex_prompt,
                    prefix=PROMPT_CONFIG["EN_REVIEW_PREFIX" ],
                    suffix=PROMPT_CONFIG["EN_REVIEW_SUFFIX" ],
                    input_variables=["text", "answer", "emotion", "reason"],
                    example_separator="\n"
                    )
    
    chain = review_prompt | model
    result = chain.invoke({"text": text, "answer": answer, "emotion": emotion, "reason": reason})
    print(f"Review: {result}")
    raw_output = result.content.strip().lower()
    feedback, confidence = extract_feedback_confidence(raw_output)
    print(f"Feedback: {feedback}")
    print(f"Confidence: {confidence}")

    return feedback, confidence

def mayor_vote(revision:dict):
    answers = [v["revised_answer"] for v in revision.values()]
    vote_count = Counter(answers)
    majority_answer = vote_count.most_common(1)[0][0]
    majority_reasons = [v["revised_reason"] for v in revision.values() if v["revised_answer"] == majority_answer]
    win_voters = [k for k,_ in revision.items() if revision[k]["revised_answer"] == majority_answer]

    return majority_answer, majority_reasons, win_voters

def getRevision(peers_review, text, emotion, answer, model):
    examples = []
    for k, v in peers_review.items():
        examples.append(
            {
                'revision': v['reviews'],
                'confidence': v['confidence']
            }
        )

    rev_prompt = PromptTemplate(
                    input_variables=['revision', 'confidence'],
                    template=PROMPT_CONFIG["REV_TEMPLATE"]
                    )
    
    revise_prompt = FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=rev_prompt,
                    prefix=PROMPT_CONFIG["REVISE_PREFIX" ].format(text=text, emotion=emotion, answer=answer),
                    suffix=PROMPT_CONFIG["REVISE_SUFFIX" ],
                    input_variables=["emotion"],
                    example_separator="\n"
                    )
    
    chain = revise_prompt | model
    result = chain.invoke({"emotion": emotion})
    raw_output = result.content.strip().lower()
    new_answer, new_reason = extract_revision(raw_output)
    return new_answer, new_reason

def summarizer(reason, model):
    combined_text = "\n".join(reason)
    summary_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Summarize the following text:\n\n{text}"
                    )
    chain = summary_prompt | model
    summaries = chain.invoke({"text": combined_text}).content
    return summaries

def multiagents_classify_text(text, model_list, fs_prompt, emotion):
    
    output_dict = {}
    revision = {}
    models =[]
    for name, model in model_list.items():
        print(f"Current model used: {name}")
        chain = fs_prompt | model
        result = chain.invoke({"input": text})
        output_text = result.content.strip().lower()
        answer = convert_output_text(output_text, 'english')
        print(f"Initial answer: {answer}")
        print(f"Initial reason: {output_text}")
        output_dict[name] = {"answer": answer,
                              "reason": output_text
                              }
        print("-"*10)

    for name, model in model_list.items():
        #cut the model from list of model
        peers = {k:v for k, v in model_list.items() if k!= name}
        peers_review = {}
        for k, v in peers.items():
            examples = fs_prompt.examples
            reviews, confidence = getPeersReview(v, output_dict[name]["answer"], 
                                                 output_dict[name]["reason"],
                                                 text,
                                                 emotion,
                                                 examples)
            peers_review[k] = {"reviews": reviews,
                               "confidence": confidence}
            print(f"Peers-review from {k} to {name} is: {reviews}")
        
        revised_answer, revised_reason = getRevision(peers_review, text, emotion, output_dict[name]["answer"], model)
        revision[name] = {"revised_answer": revised_answer,
                           "revised_reason": revised_reason}
        print(f"Revised answer for model: {name} is-- {revised_answer} from initial answer: {answer}")
        if revised_answer != answer:
            models.append(name)
        # print(f"Revised reason: {revised_reason}")
    
    #get the answer with most vote
    voting_answer, reason, win_voters = mayor_vote(revision)
    reason_summary = summarizer(reason, model_list["qwen3-32b"])
    final_result = {
                    "final_answer": voting_answer,
                    "final_reason": reason_summary,
                    "models_change_perspective": models,
                    "model_initial_answer": output_dict,
                    "model_revision": revision,
                    "winner_model": win_voters
                    }
    return final_result


def translate_label(label, prompt_language: str):
    if isinstance(label, str):
        word = label.lower()
        label_list = word.split(", ")
    else:
        label_list = label

    lang = prompt_language.lower()
    lb_str = []

    for lb in label_list:

        if lang == 'english':
            lb_str.append(lb)
        
        elif lb in TRANSLATION_MAP:
            if lang in TRANSLATION_MAP[lb]:
                lb_str.append(TRANSLATION_MAP[lb][lang])
                # return TRANSLATION_MAP[label][lang]
            else:
                return f"Error: Translation for '{lb}' to '{lang}' not found."
        else:
            return f"Error: English word '{lb}' not found in map."
    
    if isinstance(label, str):
        return ", ".join(lb_str)
    else:
        return lb_str

def translate_emotion_text(prompt_language: str):
    lang = prompt_language.lower()
    if lang == 'indonesian':
        return " Teks ini mengandung emosi: "
    elif lang == 'balinese':
        return " Tulisan puniki madaging emosi: "
    else:
        return " This text is contain emotion: "

def translate_answer(answer: str, prompt_language: str):
    ans = answer.lower()
    lang = prompt_language.lower()

    if lang == 'english':
        return ans
    elif lang == 'indonesian':
        if ans == 'yes':
            return 'ya'
        elif ans == 'no':
            return 'tidak'
    else:
        if ans == 'yes':
            return 'inggih'
        elif ans == 'no':
            return 'ten'

def llm_selection(keys:dict, llm_model_name:str = None):
    """Select and return the appropriate LLM model based on the provided name."""
    """
   
    if llm_model_name.lower() == 'gemini-2.5-flash':
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = keys.get("GOOGLE_API_KEY") or getpass.getpass("Enter your Google API Key: ")
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=None,
            timeout=None,
        )
        
        return llm_gemini
    
    elif llm_model_name.lower() == 'qwen3-32b':
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = keys.get("GROQ_API_KEY") or getpass.getpass("Enter your GROQ API Key: ")
       
        llm_qwen = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.3,
            max_output_tokens=None,
            timeout=None,
        )
        return llm_qwen

    elif llm_model_name.lower() == 'deepseek-reasoner':
        llm_deepseek = ChatOpenAI(
            model_name="deepseek-reasoner",
            openai_key=keys.get("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,
        )

        return llm_deepseek
 
    else:
        raise ValueError(f"Unsupported LLM model name: {llm_model_name}")
    """
    
    """Select and return the appropriate LLM model(s) based on the provided name.
    
    If llm_model_name is None -> return a list of all available models.
    """

    def get_google_model():
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = keys.get("GOOGLE_API_KEY") or getpass.getpass(
                "Enter your Google API Key: "
            )
        llm_model =  ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            timeout=None,
        )
        return {"gemini-2.5-flash": llm_model}

    def get_qwen_model():
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = keys.get("GROQ_API_KEY") or getpass.getpass(
                "Enter your GROQ API Key: "
            )
        llm_model =  ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.3,
        )
        return {"qwen3-32b": llm_model}
    
    def get_llama_model():
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = keys.get("GROQ_API_KEY") or getpass.getpass(
                "Enter your GROQ API Key: "
            )
        llm_model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        return {"llama-3.3-70b": llm_model}

    def get_deepseek_model():
        llm_model = ChatOpenAI(
            model_name="deepseek-reasoner",
            openai_api_key=keys.get("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,
        )
        return {"deepseek-reasoner": llm_model}
    
    def get_kimik2_model():
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = keys.get("GROQ_API_KEY") or getpass.getpass(
                "Enter your GROQ API Key: "
            )
        llm_model = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905",
            temperature=0.3,
        )
        return {"kimi-k2-instruct-0905": llm_model}

    # --- Return all models case ---
    if llm_model_name is None:
        all_models = {}
        #all_models.update(get_google_model())
        all_models.update(get_kimik2_model())
        all_models.update(get_qwen_model())
        all_models.update(get_llama_model())
        #all_models.update(get_deepseek_model())
        return all_models

    # --- Single model selection ---
    name = llm_model_name.lower()

    if name == "gemini-2.5-flash":
        return get_google_model()
    elif name == "qwen3-32b":
        return get_qwen_model()
    elif name == "deepseek-reasoner":
        return get_deepseek_model()
    elif name == "llama-3.3-70b":
        return get_llama_model()
    elif name == "kimi-k2-instruct-0905":
        return get_kimik2_model()
    else:
        raise ValueError(f"Unsupported LLM model name: {llm_model_name}")
    
def strip_result_content(content, conf_answer, conf_reason):
    temp_content = content.split(f"{conf_answer}: ")[1]
    parts = temp_content.split(f"\n{conf_reason}:")

    if len(parts) < 2:

        answer = parts[0].strip()
        reason = "no reason" 
    else:
        answer = parts[0].strip() 
        reason = parts[1].strip()


    return answer, reason