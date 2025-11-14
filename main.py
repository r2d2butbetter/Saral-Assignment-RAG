from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL

question = "What are the key findings about training loop development?"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

docs = retriever.invoke(question)

context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
result = llm.invoke(prompt)

print(result.content)
print("\n--- Source Documents ---")
for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")