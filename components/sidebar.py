# sidebar.py

import streamlit as st

def sidebar_ui():
    st.sidebar.title("Legal Doc Analyzer")
    st.sidebar.markdown("""
    Upload Indian legal documents (PDF or Text) for analysis.  
    Features include:  
    - Simple summary generation  
    - Glossary extraction with explanations  
    - Interactive Q&A about the document  
    """)
    
    # File uploader moved here if you want
    uploaded_file = st.sidebar.file_uploader("Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])

    # Optional: Select difficulty level filter for glossary terms
    difficulty_filter = st.sidebar.multiselect(
        "Filter Glossary Terms by Difficulty",
        options=["Easy", "Medium", "Hard"],
        default=["Easy", "Medium", "Hard"]
    )

    # Optional: Theme toggle or settings could go here

    return uploaded_file, difficulty_filter
