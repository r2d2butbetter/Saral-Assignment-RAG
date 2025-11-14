from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL


embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

template = """
You are a helpful assistant. Use the following context to answer the user's request.
Context:
{context}

---
Task: Perform the following request.
Request: {question}
Audience: {audience}
Length: {length}
Style: {style}

Format:
1.  **Slide Bullets:** A list of bullet points for slides.
2.  **Speaker Script:** A full speaker script.
3.  **Speaker Notes:** 3 notes per slide, preserving key equations in LaTeX.

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": (lambda x: x["question"]),
        "audience": (lambda x: x["audience"]),
        "length": (lambda x: x["length"]),
        "style": (lambda x: x["style"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

input_data = {
    "question": "What are the key findings about training loop development?",
    "audience": "5 year old kids",
    "length": "30 second script",
    "style": "Policymakers"
}

print(f"Generating response: '{input_data['question']}'")
result = rag_chain.invoke(input_data)

print("\n\n")
print(result)


print("\n\n Source docs")
docs = retriever.invoke(input_data["question"])
for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")