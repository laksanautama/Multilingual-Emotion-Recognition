
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import DATA_CONFIG
from utils import llm_selection, translate_label, translate_emotion_text
import os
from datasets import Dataset
from langchain_classic.chains import RetrievalQA


def store_vectors_database(train_data, keys: dict):
    """Placeholder function to store vectors in the database."""
    print("Storing vectors in the FAISS vector DB...")
    # os.environ["GOOGLE_API_KEY"] = keys.get("GOOGLE_API_KEY")
    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    full_path = os.path.join(MAIN_DIR, DATA_CONFIG["FAISS_INDEX_DIR"])
    
    # train_data = train_data.reset_index()
    # train_data = train_data.rename(columns={'index': 'id'})

    documents = []

    hf_traindata = Dataset.from_dict(
    {
        'id': train_data['id'],
        'text': train_data['text'],
        'emotions': train_data['emotions']
    })

    for data in hf_traindata:
        documents.append(
            Document(
            page_content=data['text'],
            metadata={"id": data['id'],
                      "emotions": data['emotions']}
            )
        )
    embeddings = GoogleGenerativeAIEmbeddings(
        model = DATA_CONFIG["EMBEDDING_MODEL"]
        )
    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_store.save_local(f"{full_path}")
    print(f"Vectors stored successfully to {full_path}.")


def rag_retriever(llm_model_name: str, keys:dict, k: int):
  MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
  full_path = os.path.join(MAIN_DIR, DATA_CONFIG["FAISS_INDEX_DIR"])
  embeddings = GoogleGenerativeAIEmbeddings(
              model = DATA_CONFIG["EMBEDDING_MODEL"]
            )

  vectorstore = FAISS.load_local(f"{full_path}", 
                                    embeddings,
                                    allow_dangerous_deserialization=True
                                    )
    
  retriever = vectorstore.as_retriever(search_kwargs={"k": k})
  return retriever

def docs_retrieval_processing(docs, lang):
  doc_list = []
  for doc in docs:
    if isinstance(doc.metadata['emotions'], list):
      emotions_list = [e.strip() for e in doc.metadata['emotions'] if e.strip()]
      str_emotion = ", ".join(emotions_list)
    else:
      str_emotion = doc.metadata['emotions'].strip("[]'").replace("'", "")
    
    en_emotion = 'no emotion' if str_emotion == '' else str_emotion

    emotion = translate_label(en_emotion, lang)
    translated_text = translate_emotion_text(lang)
    text = f"{translated_text}{emotion}"
    complete_text = doc.page_content + text
    doc_list.append(complete_text)
  context = "\n\n".join([x for x in doc_list])
  return context

def rag_classifier(query, retriever, chain, labels, lang):
  """
    -- you need to convert from label in eng to ind or bal
    -- you also need to translate 'This text is contain emotion'
  """
  docs = retriever.invoke(query)
  context = docs_retrieval_processing(docs, lang)
  labels.append('no emotion') 
  translated_label = []
  for lab in labels:
     tr_label = translate_label(lab, lang)
     translated_label.append(tr_label)

  labels_text = ", ".join(translated_label)

  input_dict = {
        "context": context,
        "query": query,
        "labels": labels_text
        }
  result = chain.invoke(input_dict)
        
  return result

# def retrieve_docs_for_emotion(query, emotion, pos_retriever, neg_retriever):
#     pos_docs = pos_retriever.invoke(query)
#     neg_docs = neg_retriever.invoke(query)
#     return pos_docs + neg_docs

def manual_retriever(retriever, query, target_emotion):
  pool = retriever.invoke(query)
  pos_docs = []
  neg_docs = []

  for doc in pool:
    doc_emotions = doc.metadata.get("emotions", [])
    
    if isinstance(doc_emotions, str):
      doc_emotions = [e.strip() for e in doc_emotions.strip("[]").split(",")]

    if target_emotion in doc_emotions:
      if len(pos_docs) < DATA_CONFIG["RAG_TOP_K"]:
        pos_docs.append(doc)
    else:
      if len(neg_docs) < DATA_CONFIG["RAG_TOP_K"]:
        neg_docs.append(doc)

    if len(pos_docs) >= DATA_CONFIG["RAG_TOP_K"] and len(neg_docs) >= DATA_CONFIG["RAG_TOP_K"]:
      break

  return pos_docs + neg_docs


def bin_rag_classifier(query, retriever, chain, target_emotion, lang):
    # docs = retrieve_docs_for_emotion(query, target_emotion, pos_retriever, neg_retriever)
    docs = manual_retriever(retriever, query, target_emotion)
    context = docs_retrieval_processing(docs, lang)

    # Translate the single label
    translated_label = translate_label(target_emotion, lang)

    input_dict = {
        "context": context,
        "query": query,
        "target_label": translated_label
    }

    result = chain.invoke(input_dict)
    return result

