from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL
from models import FullScriptUpdate, SlideData


def format_docs_with_citations(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source_info = f"[Source {i}: Page {doc.metadata.get('page', 'N/A')}]"
        formatted.append(f"{source_info}\n{doc.page_content}")
    return "\n\n".join(formatted)


def load_models_and_retriever():
    print("Loading models and vector store")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    
    #llm for chatting
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
    
    # llm for json output
    json_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
    structured_llm = json_llm.with_structured_output(FullScriptUpdate)

    print("Models and vector store loaded.")
    return llm, retriever, structured_llm


def contextualize_question(llm, inputs):
    chat_history = inputs.get("chat_history", [])
    input_text = inputs["input"]
    
    if not chat_history:
        return input_text
    
    #prompt to add context
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # mini chain
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()
    return contextualize_chain.invoke({
        "chat_history": chat_history,
        "input": input_text
    })


def create_rag_chain(llm, retriever):
    # Main prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful assistant for the SARAL project. Use the following context to answer the user's request.
Context:
{context}

Task: Perform the following request.
Request: {input}

If the user asks for a script, bullets, or abstract, please provide it in a clear format.

CRITICAL FORMATTING RULES:
1. ALL mathematical equations MUST be written in proper LaTeX format
2. Use $...$ for inline math (e.g., $L_G = L_{{adversarial}} + \\lambda L_{{L1}}$)
3. Use $$...$$ for display/block equations
4. NEVER use HTML tags like <sub>, <sup>, or Unicode subscripts/superscripts
5. For subscripts use underscore: $L_{{adversarial}}$ not L<sub>adversarial</sub>
6. For superscripts use caret: $x^2$ not x²

IMPORTANT: When using information from the context, cite the source inline using the format [Source X] where X is the source number provided in the context.
Only cite sources that you actually use in your response.

Do not add any special tokens like <end_message> or <|end|> to your response.
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])



    #lcel part
    rag_chain = (
        {
            "context": lambda x: format_docs_with_citations(retriever.invoke(contextualize_question(llm, x))),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def create_refinement_chain(llm, retriever, structured_llm):
    
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert script editor. Your task is to refine a previous script based on a user's request.
You must provide your output in the requested JSON format.

Here is the previous script:
---
{last_output}
---

Here is additional context retrieved from the knowledge base:
---
{context}
---

Here is the user's refinement request:
---
{input}
---

Use the retrieved context to add accurate details. Identify the changes, provide the old and new text, explain the reason, and give the full updated script.

CRITICAL FORMATTING RULES:
1. ALL mathematical equations MUST be written in proper LaTeX format
2. Use $...$ for inline math (e.g., $L_G = L_{{adversarial}} + \\lambda L_{{L1}}$)
3. Use $$...$$ for display/block equations
4. NEVER use HTML tags like <sub>, <sup>, or Unicode subscripts/superscripts
5. For subscripts use underscore: $L_{{adversarial}}$ not L<sub>adversarial</sub>
6. For superscripts use caret: $x^2$ not x²

IMPORTANT: When adding information from the context, cite the source inline using the format [Source X] where X is the source number provided in the context.
Only cite sources that you actually use in your response.

Do not add any special tokens like <end_message> or <|end|> to your response.
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # lcel chain
    refinement_chain = (
        {
            "context": lambda x: format_docs_with_citations(retriever.invoke(contextualize_question(llm, x))),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
            "last_output": lambda x: x["last_output"]
        }
        | refine_prompt
        | structured_llm
    )
    
    return refinement_chain


def create_slide_chain(llm, retriever):
    """Chain to generate ppt from prompt"""    
    #structured llm for slides
    json_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
    slide_structured_llm = json_llm.with_structured_output(SlideData)
    
    slide_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert presentation designer. Create a professional presentation based on the provided context.

Context:
{context}

User Request:
{input}

Generate a presentation with:
1. A clear, engaging presentation title
2. Multiple slides (typically 4-8 slides)
3. Each slide should have:
   - A concise, descriptive title
   - 2-3 bullet points that are clear and informative
   - Bullet points should be concise and short (1 liner max)

Make sure the content is derived from the provided context and addresses the user's request.
Keep bullet points professional and to the point.
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    slide_chain = (
        {
            "context": lambda x: format_docs_with_citations(retriever.invoke(contextualize_question(llm, x))),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | slide_prompt
        | slide_structured_llm
    )
    
    return slide_chain
