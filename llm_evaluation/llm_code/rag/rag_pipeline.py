
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
import os


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
            metadata={"id": row['id']}
            )
        )
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "text-embedding-004"
        )
    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_store.save_local(f"{full_path}")
    print(f"Vectors stored successfully to {full_path}.")