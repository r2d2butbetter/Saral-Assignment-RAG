from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

loader = PyMuPDFLoader("sample_docs/report.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local("faiss_index")
