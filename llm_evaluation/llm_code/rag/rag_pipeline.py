
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from llm_evaluation.llm_code.llm_utility.llm_utility import llm_selection
import os
from langchain_classic.chains import RetrievalQA


def store_vectors_database(train_data, keys: dict):
    """Placeholder function to store vectors in the database."""
    print("Storing vectors in the FAISS vector DB...")
    # os.environ["GOOGLE_API_KEY"] = keys.get("GOOGLE_API_KEY")
    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    full_path = os.path.join(MAIN_DIR, DATA_CONFIG["FAISS_INDEX_DIR"])
    
    train_data = train_data.reset_index()
    train_data = train_data.rename(columns={'index': 'id'})

    documents = []
    for _, row in train_data.iterrows():
        documents.append(
            Document(
            page_content=row['sentence'],
            metadata={"id": row['id'],
                      "emotions": row['emotions']}
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
    # llm = llm_selection(llm_model_name, keys)
    
    # rag = RetrievalQA.from_chain_type(  
    #     llm=llm,
    #     retriever=retr,
    #     return_source_documents=False
    # )
    return retriever

def docs_retrieval_processing(docs):
  doc_list = []
  for doc in docs:
    str_emotion = doc.metadata['emotion'].strip("[]'").replace("'", "")
    emotion = 'no emotion' if str_emotion == '' else str_emotion
    text = f" This text is contain emotion: {emotion}"
    complete_text = doc.page_content + text
    doc_list.append(complete_text)
  context = "\n\n".join([x for x in doc_list])
  return context

def rag_classifier(query, retriever, chain, labels):
  docs = retriever.invoke(query)
  context = docs_retrieval_processing(docs)
  labels.append('no emotion') #adding 'no emotion' label
  labels_text = ", ".join(labels)

  input_dict = {
        "context": context,
        "query": query,
        "labels": labels_text
        }
  result = chain.invoke(input_dict)
        
  return result
