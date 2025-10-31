import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def classify_text(text, chain):
    """Classify the input text using the provided LLM chain."""
    result = chain.invoke({"input": text})
    output_text = result.content.strip().lower()
    if '\n\nanswer: yes' in output_text:
        return 'yes'
    elif '\n\nanswer: no' in output_text:
        return 'no'
    else:
    # Return a default or handle cases where neither 'yes' nor 'no' is found
        return 'no'
    

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