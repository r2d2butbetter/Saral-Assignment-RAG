from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

docs = retriever.invoke('development of training loop')

print(f"Number of documents retrieved: {len(docs)}")
print(docs[0])


