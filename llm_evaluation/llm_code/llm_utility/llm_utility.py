import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from .llm_config import PROMPT_CONFIG, TRANSLATION_MAP
from crosslingual_ER.scripts.model_configs import DATA_CONFIG


def select_language_config(prompt_language: str):
    """Select the appropriate prompt configuration based on the specified language."""

    if prompt_language.lower() == 'indonesian' or prompt_language.lower() == 'ind':
        
        config = {
                    'example_template' : PROMPT_CONFIG["IND_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["IND_PREFIX"],
                    'suffix' : PROMPT_CONFIG["IND_SUFFIX"],
                    'clf_template': PROMPT_CONFIG["IND_CLF_TEMPLATE"],
                    'label_column' : DATA_CONFIG["IND_LABELS"],
                    'input' : "teks_masukan",
                    'answer' : "jawaban",
                    'reason' : "alasan"

                    }
        
    elif prompt_language.lower() == 'balinese' or prompt_language.lower() == 'bal':

        config = {
                    'example_template' : PROMPT_CONFIG["BAL_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["BAL_PREFIX"],
                    'suffix' : PROMPT_CONFIG["BAL_SUFFIX"],
                    'clf_template' : PROMPT_CONFIG["BAL_CLF_TEMPLATE"],
                    'label_column' : DATA_CONFIG["BAL_LABELS"],
                    'input' : "masukan",
                    'answer' : "pasaut",
                    'reason' : "alasan"

                    }

    else:  # Default to English
        
        config = {
                    'example_template' : PROMPT_CONFIG["EN_EXAMPLE_TEMPLATE"],
                    'prefix' : PROMPT_CONFIG["EN_PREFIX"],
                    'suffix' : PROMPT_CONFIG["EN_SUFFIX"],
                    'clf_template' : PROMPT_CONFIG["CLF_TEMPLATE"],
                    'label_column' : DATA_CONFIG["LABELS"],
                    'input' : "input",
                    'answer' : "answer",
                    'reason' : "reason"

                    }

    return config


def convert_output_text(output_text: str, prompt_language: str):
    """Convert the output text from the LLM to standardized 'yes' or 'no'."""
    output_text = output_text.strip().lower()
    if prompt_language.lower() == 'indonesian' or prompt_language.lower() == 'ind':
        if '\n\njawaban: ya' in output_text:
            return 'yes'
        elif '\n\njawaban: tidak' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'ya' nor 'tidak' is found
            return 'no'
    elif prompt_language.lower() == 'balinese' or prompt_language.lower() == 'bal':
        if '\n\npasaut: inggih' in output_text:
            return 'yes'
        elif '\n\npasaut: ten' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'inggih' nor 'ten' is found
            return 'no'
        
    else:  # Default to English
        if '\n\nanswer: yes' in output_text:
            return 'yes'
        elif '\n\nanswer: no' in output_text:
            return 'no'
        else:
            # Return a default or handle cases where neither 'yes' nor 'no' is found
            return 'no'

def classify_text(text, chain, prompt_language: str):
    """Classify the input text using the provided LLM chain."""
    result = chain.invoke({"input": text})
    output_text = result.content.strip().lower()

    print(f"=== {output_text} ===")

    return convert_output_text(output_text, prompt_language), output_text

def translate_label(label: str, prompt_language: str):
    word = label.lower()
    lang = prompt_language.lower()

    if lang == 'english':
        return word
    
    if word in TRANSLATION_MAP:
        if lang in TRANSLATION_MAP[word]:
            return TRANSLATION_MAP[word][lang]
        else:
            return f"Error: Translation for '{word}' to '{lang}' not found."
    else:
        return f"Error: English word '{word}' not found in map."

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

def llm_selection(llm_model_name: str, keys: dict):
    """Select and return the appropriate LLM model based on the provided name."""
   
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