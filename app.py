import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from chains import load_models_and_retriever, create_rag_chain, create_refinement_chain, create_slide_chain, contextualize_question
from display import display_changes, display_answer, display_sources, display_slides


#loading models.
@st.cache_resource
def get_models_and_chains():
    llm, retriever, structured_llm = load_models_and_retriever()
    rag_chain = create_rag_chain(llm, retriever)
    refinement_chain = create_refinement_chain(llm, retriever, structured_llm)
    slide_chain = create_slide_chain(llm, retriever)
    return llm, retriever, rag_chain, refinement_chain, slide_chain

llm, retriever, rag_chain, refinement_chain, slide_chain = get_models_and_chains()


st.set_page_config(page_title="SARAL Chatbot Prototype", layout="wide")
st.title("SARAL Chatbot Prototype")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_output" not in st.session_state:
    st.session_state.last_output = ""

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("ai").write(msg.content)

# Slide generation toggle
generate_slides = st.checkbox("📊 Also generate presentation slides", value=False)

if prompt := st.chat_input():

    st.chat_message("user").write(prompt)
    
    is_refinement_request = bool(st.session_state.last_output)
    
    with st.chat_message("ai"):
        with st.spinner("Loading"):
            
            if is_refinement_request:
                st.write("Refining previous output with additional context from knowledge base...")
                
                try:
                    contextualized_question = contextualize_question(llm, {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    retrieved_docs = retriever.invoke(contextualized_question)
                    
                    update_object = refinement_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "last_output": st.session_state.last_output
                    })
                    
                    display_changes(update_object)

                    st.divider()
                    st.subheader("Full Updated Script:")
                    
                    answer = update_object.full_updated_script
                    
                    display_answer(answer)
                    
                    display_sources(answer, retrieved_docs, "Refinement")

                except Exception as e:
                    st.error(f"Error during refinement: {e}")
                    answer = "I tried to refine that, but ran into an error."

            else:
                st.write("Generating new output...")
                
                contextualized_question = contextualize_question(llm, {
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                retrieved_docs = retriever.invoke(contextualized_question)
                
                answer = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                display_answer(answer)
                display_sources(answer, retrieved_docs)
                
                # Generate slides if requested
                if generate_slides:
                    st.divider()
                    with st.spinner("Generating presentation slides..."):
                        try:
                            slide_data = slide_chain.invoke({
                                "input": prompt,
                                "chat_history": st.session_state.chat_history
                            })
                            display_slides(slide_data)
                        except Exception as e:
                            st.error(f"Error generating slides: {e}")

    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.last_output = answer # Save for next refinement

