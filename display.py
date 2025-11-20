import streamlit as st
import difflib
from utils import extract_cited_sources, create_presentation


def display_changes(update_object):
    st.subheader("Changes Made:")
    for change in update_object.changes:
        st.text(f"Reason: {change.reason_for_change}")
        
        # Calculate and display diff
        diff = difflib.unified_diff(
            change.old_text_segment.splitlines(),
            change.new_text_segment.splitlines(),
            fromfile="Old",
            tofile="New",
            lineterm="",
        )
        st.code('\n'.join(diff), language="diff")


def display_answer(answer):
    st.markdown(answer)


def display_sources(answer, retrieved_docs, source_type=""):
    cited_sources = extract_cited_sources(answer)
    
    if cited_sources:
        source_label = f"View {len(cited_sources)} Cited Source(s)"
        if source_type:
            source_label = f"View {len(cited_sources)} Cited Source(s) from {source_type}"
        
        with st.expander(f"📚 {source_label}"):
            for source_num in sorted(cited_sources, key=int):
                idx = int(source_num) - 1
                if idx < len(retrieved_docs):
                    doc = retrieved_docs[idx]
                    st.write(f"**Source {source_num} (Page {doc.metadata.get('page', 'N/A')})**")
                    st.write(doc.page_content)
                    st.divider()


def display_slides(slide_data):
    """Display slide preview and provide download button."""
    st.subheader("📊 Presentation Slides")
    
    # Show preview
    with st.expander(f"Preview: {slide_data.presentation_title} ({len(slide_data.slides)} slides)", expanded=True):
        st.markdown(f"### {slide_data.presentation_title}")
        st.divider()
        
        for i, slide in enumerate(slide_data.slides, 1):
            st.markdown(f"**Slide {i}: {slide.title}**")
            for bullet in slide.bullets:
                st.markdown(f"- {bullet}")
            if i < len(slide_data.slides):
                st.divider()
    
    # Generate PowerPoint file
    try:
        pptx_buffer = create_presentation(slide_data)
        st.download_button(
            label="📥 Download PowerPoint",
            data=pptx_buffer,
            file_name=f"{slide_data.presentation_title.replace(' ', '_')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    except Exception as e:
        st.error(f"Error generating PowerPoint: {e}")
