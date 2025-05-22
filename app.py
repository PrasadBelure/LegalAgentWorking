import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import google.generativeai as genai
import re
import os
import difflib
import pandas as pd
from datetime import datetime
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import io
import tempfile
import torch

# Download required NLTK data
# Add the local nltk_data folder to NLTK's search path
nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_path)

# Download only if missing (won't usually run on Streamlit)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_path, quiet=True)

# --- App Configuration ---
st.set_page_config(
    page_title="LexScanCite  - AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
    }
    .info-box {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: rgba(239, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
    }
    .legal-term {
        text-decoration: underline;
        color: #4B5563;
        cursor: help;
        position: relative;
    }
    .legal-term:hover::after {
        content: attr(data-explanation);
        position: absolute;
        left: 0;
        top: 100%;
        min-width: 200px;
        max-width: 400px;
        background: rgba(243, 244, 246, 0.95);
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #D1D5DB;
        z-index: 10;
        font-weight: normal;
        color: #1F2937;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .highlight {
        background-color: rgba(254, 252, 232, 0.7);
        padding: 0.25rem;
        border-radius: 0.25rem;
    }
    .diff-added {
        background-color: rgba(236, 253, 245, 0.7);
        padding: 0.25rem;
        border-radius: 0.25rem;
        border-left: 3px solid #10B981;
    }
    .diff-removed {
        background-color: rgba(254, 242, 242, 0.7);
        padding: 0.25rem;
        border-radius: 0.25rem;
        border-left: 3px solid #EF4444;
        text-decoration: line-through;
    }
    .search-result {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(209, 213, 219, 0.5);
        background-color: rgba(249, 250, 251, 0.1);
    }
    .search-highlight {
        background-color: rgba(252, 211, 77, 0.3);
        padding: 0 0.25rem;
        border-radius: 0.25rem;
    }
    /* Fix for white text on white background */
    .stApp {
        color: inherit !important;
    }
    /* Add more contrast to buttons */
    .stButton>button {
        color: white;
        background-color: #3B82F6;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini API Setup ---
def initialize_genai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        # Fallback to direct API key if not in secrets
        api_key = "AIzaSyAxwVvINTcwBuWD0ILpWp0LuU7GFhb0CTE"  # Replace with your API key
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

model = initialize_genai()

# --- Embedding Model Initialization ---
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model for embeddings"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'legal_terms_db': pd.DataFrame(columns=['term', 'explanation']),
        'document_content': None,
        'document_summary': None,
        'document_hash': None,
        'comparison_content': None,
        'comparison_results': None,
        'document_sentences': None,
        'sentence_embeddings': None,
        'faiss_index': None,
        'current_question': "",
        'embeddings_processed': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Text Extraction ---
def extract_text_from_pdf(file) -> str:
    try:
        # Save the BytesIO to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        
        # Open the temporary file with PyMuPDF
        doc = fitz.open(temp_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Close and delete the temporary file
        doc.close()
        try:
            os.unlink(temp_file_path)
        except:
            pass  # In case file deletion fails
            
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text(file):
    if file is None:
        return ""
    
    file_type = file.type
    if "pdf" in file_type:
        return extract_text_from_pdf(file)
    else:
        return file.getvalue().decode("utf-8")

# --- Document Preprocessing ---
def preprocess_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean up common PDF extraction artifacts
    text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenated words
    return text.strip()

# --- Text Embedding and FAISS Index Creation ---
def create_text_embeddings(text):
    """Create embeddings for document text using sentence embeddings"""
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return None, None
    
    try:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Generate embeddings for each sentence
        sentence_embeddings = embedding_model.encode(sentences)
        
        # Create FAISS index
        dimension = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(sentence_embeddings).astype('float32'))
        
        return sentences, sentence_embeddings, index
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None, None, None

# --- Semantic Search Function ---
def semantic_search(query, top_k=5):
    """Perform semantic search on document text"""
    if (st.session_state.sentence_embeddings is None or 
        st.session_state.faiss_index is None or 
        st.session_state.document_sentences is None):
        return []
    
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return []
    
    try:
        # Generate embedding for query
        query_embedding = embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = st.session_state.faiss_index.search(
            np.array(query_embedding).astype('float32'), top_k
        )
        
        # Get corresponding sentences
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(st.session_state.document_sentences):
                results.append({
                    'sentence': st.session_state.document_sentences[idx],
                    'distance': distances[0][i],
                    'index': idx
                })
        
        return results
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")
        return []

# --- AI-Powered Analysis Functions ---
def generate_summary(text, level="Standard"):
    """Generate document summary with specified detail level"""
    detail_map = {
        "Concise": "brief (1 paragraph)",
        "Standard": "moderate length (2-3 paragraphs)",
        "Detailed": "comprehensive (4-5 paragraphs)"
    }
    
    # Limit text length to stay within token limits
    max_chars = 12000  # Adjust based on model limitations
    input_text = text[:max_chars] + ("..." if len(text) > max_chars else "")
    
    prompt = f"""
You are an expert legal document analyst specializing in Indian law.
Please provide a {detail_map[level]} summary of the following legal document that would be helpful for both lawyers and laypeople.
Focus on:
1. The key purpose/subject of the document
2. Main rights and obligations established
3. Important dates or deadlines
4. Key parties involved
5. Notable conditions or requirements

Document Text:
{input_text}

Summary:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def explain_legal_term(term):
    """Get explanation for a legal term"""
    prompt = f"""
As an expert Indian legal professional, explain the legal term "{term}" in simple language.
Provide:
1. A clear definition that a non-lawyer can understand (1-2 sentences)
2. The legal context or importance of this term
3. Any relevant information about how this term is used in Indian law specifically

Keep the total explanation under 150 words and make it accessible to someone without legal training.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error explaining term: {str(e)}"

def compare_documents(text1, text2):
    """Compare two legal documents and identify key differences"""
    # Prepare prompts with text samples (handling length limits)
    max_chars = 10000  # Limiting input size
    text1_sample = text1[:max_chars] + ("..." if len(text1) > max_chars else "")
    text2_sample = text2[:max_chars] + ("..." if len(text2) > max_chars else "")
    
    prompt = f"""
As a legal document comparison expert, analyze these two legal documents and identify the important differences.
Focus on material differences that would matter to someone reviewing these documents, such as:

1. Changes in rights, obligations, or liabilities
2. Added or removed terms, conditions, or clauses
3. Changes in dates, deadlines, or timeframes
4. Modifications to requirements or processes
5. Changes in defined terms or key parties

Document 1:
{text1_sample}

Document 2:
{text2_sample}

Provide your analysis in this format:
1. SUMMARY: A brief summary of the most significant differences (2-3 sentences)
2. KEY_DIFFERENCES: A numbered list of the most important substantive differences
3. ADDED_IN_DOC2: Important elements present only in Document 2
4. REMOVED_IN_DOC2: Important elements present only in Document 1
5. LEGAL_IMPLICATIONS: Potential legal implications of these changes

Be specific about where differences appear when possible.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error comparing documents: {str(e)}"

def analyze_document_qa(text, query):
    """Answer specific questions about the legal document"""
    max_chars = 12000
    text_sample = text[:max_chars] + ("..." if len(text) > max_chars else "")
    
    # Get semantic search results to include as context
    search_results = semantic_search(query, top_k=5)
    context_text = ""
    
    if search_results:
        context_text = "Relevant document sections:\n\n"
        for i, result in enumerate(search_results):
            context_text += f"{i+1}. {result['sentence']}\n"
    
    prompt = f"""
As an expert legal analyst, answer this specific question about the legal document provided.
Base your answer primarily on the information present in the document text, with special attention to the relevant sections I've identified.
If the answer cannot be determined from the text, clearly state this.

{context_text}

Full Document Text:
{text_sample}

Question: {query}

Provide a clear, direct answer with references to specific parts of the document when relevant.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error answering question: {str(e)}"

def extract_legal_terms(text):
    """Extract potential legal terms from document text"""
    prompt = f"""
As a legal expert, identify 10 important legal terms or phrases that appear in the following document.
Choose terms that would be helpful to explain to someone without legal training.
Only list terms that actually appear in the document text.

Document Text:
{text[:10000]}

Provide only the list of terms, one per line, nothing else.
"""
    
    try:
        response = model.generate_content(prompt)
        terms = response.text.strip().split('\n')
        return [term.strip() for term in terms if term.strip()]
    except Exception as e:
        st.error(f"Error extracting legal terms: {e}")
        return []

# --- Sidebar UI ---
def sidebar_ui():
    st.sidebar.markdown('<h1 style="text-align: center;">LexScanCite</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="text-align: center;">Your AI Legal Document Assistant</p>', unsafe_allow_html=True)
    
    with st.sidebar.expander("ℹ️ About the App", expanded=False):
        st.markdown("""
        **LexScanCited Pro** is a powerful legal document analysis tool that helps you:
        
        - Generate clear summaries of complex legal documents
        - Understand legal terminology with on-demand explanations
        - Perform semantic search within documents
        - Compare documents to identify key differences
        - Ask specific questions about document content
        
        Perfect for lawyers, business professionals, and individuals working with legal documents.
        """)
    
    uploaded_file = st.sidebar.file_uploader("📄 Upload Legal Document (PDF or TXT)", type=["pdf", "txt"])
    
    document_type = st.sidebar.selectbox(
        "📑 Document Type",
        options=[
            "Contract/Agreement", 
            "Court Judgment", 
            "Legislation/Act", 
            "Legal Notice", 
            "Regulatory Document",
            "Other"
        ]
    )
    
    jurisdiction = st.sidebar.selectbox(
        "🏛️ Jurisdiction",
        options=[
            "India (Central)", 
            "State Laws", 
            "International", 
            "Not Specified"
        ]
    )
    
    with st.sidebar.expander("⚙️ Summary Settings"):
        summarization_level = st.select_slider(
            "Summary Detail Level",
            options=["Concise", "Standard", "Detailed"],
            value="Standard"
        )
    
    # Second document for comparison
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Document Comparison")
    comparison_file = st.sidebar.file_uploader("📄 Upload Second Document (Optional)", type=["pdf", "txt"])
    
    return uploaded_file, document_type, jurisdiction, summarization_level, comparison_file

# --- UI Components ---
def display_document_summary(text, document_type, summarization_level):
    """Display the document summary section"""
    st.markdown("<h3 class='feature-header'>📄 Document Summary</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate document summary if not in session state
        if not st.session_state.document_summary:
            with st.spinner("Generating document summary..."):
                st.session_state.document_summary = generate_summary(text, summarization_level)
        
        st.markdown(f"<div class='info-box'>{st.session_state.document_summary}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Document Info</h4>", unsafe_allow_html=True)
        
        # Basic document stats
        word_count = len(text.split())
        
        st.markdown(f"""
        - **Type:** {document_type}
        - **Length:** ~{word_count} words
        - **Summary Level:** {summarization_level}
        """)
        
        # Add button to regenerate summary with different detail level
        if st.button("Regenerate Summary"):
            with st.spinner("Regenerating summary..."):
                st.session_state.document_summary = generate_summary(text, summarization_level)
                st.rerun()

def display_legal_terms_lookup(document_text):
    """Display the legal terms lookup section"""
    st.markdown("<h3 class='feature-header'>📚 Legal Terms Lookup</h3>", unsafe_allow_html=True)
    
    # Auto-extract legal terms option
    with st.expander("🔎 Auto-extract Legal Terms", expanded=False):
        st.markdown("Let the AI identify important legal terms from your document automatically.")
        if st.button("Extract Legal Terms"):
            with st.spinner("Identifying legal terms in document..."):
                extracted_terms = extract_legal_terms(document_text)
                
                if extracted_terms:
                    st.success(f"Found {len(extracted_terms)} potential legal terms!")
                    
                    for term in extracted_terms:
                        # Check if term already exists in database
                        if term.lower() not in st.session_state.legal_terms_db['term'].str.lower().values:
                            with st.spinner(f"Looking up '{term}'..."):
                                explanation = explain_legal_term(term)
                                
                                # Add to database
                                new_row = pd.DataFrame([{'term': term, 'explanation': explanation}])
                                st.session_state.legal_terms_db = pd.concat([st.session_state.legal_terms_db, new_row], ignore_index=True)
                    
                    st.success("Terms added to your database!")
                    st.rerun()
                else:
                    st.warning("No legal terms were identified in the document.")
    
    # Input for new term lookup
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_term = st.text_input("Enter a legal term to explain:", placeholder="e.g. habeas corpus, tort, summary judgment")
    
    with col2:
        st.write("")
        st.write("")
        lookup_clicked = st.button("Lookup Term", use_container_width=True)
    
    # Process term lookup
    if lookup_clicked and new_term:
        # Check if term already exists in database
        if new_term.lower() in st.session_state.legal_terms_db['term'].str.lower().values:
            st.info(f"'{new_term}' is already in your terms database.")
        else:
            with st.spinner(f"Looking up '{new_term}'..."):
                explanation = explain_legal_term(new_term)
                
                # Add to database
                new_row = pd.DataFrame([{'term': new_term, 'explanation': explanation}])
                st.session_state.legal_terms_db = pd.concat([st.session_state.legal_terms_db, new_row], ignore_index=True)
                
                st.success(f"Added '{new_term}' to your terms database!")
    
    # Display existing terms database
    if not st.session_state.legal_terms_db.empty:
        st.markdown("### Your Legal Terms Database")
        
        # Interactive table to show terms
        for _, row in st.session_state.legal_terms_db.iterrows():
            with st.expander(f"**{row['term']}**"):
                st.markdown(row['explanation'])
        
        # Option to clear database
        if st.button("Clear Terms Database"):
            st.session_state.legal_terms_db = pd.DataFrame(columns=['term', 'explanation'])
            st.success("Terms database cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Highlight matching terms in document
    if document_text and not st.session_state.legal_terms_db.empty:
        st.markdown("<h4>Term Highlighting in Document</h4>", unsafe_allow_html=True)
        st.markdown("The following preview shows instances of your saved legal terms highlighted in the document:")
        
        # Create document preview with highlighted terms
        preview_text = document_text[:5000] + ("..." if len(document_text) > 5000 else "")
        highlighted_text = preview_text
        
        for _, row in st.session_state.legal_terms_db.iterrows():
            term = row['term']
            explanation = row['explanation'].replace('"', '&quot;').replace("'", "&#39;")
            
            # Use regex to find whole word matches only
            pattern = r'\b' + re.escape(term) + r'\b'
            replacement = f'<span class="legal-term" data-explanation="{explanation}">{term}</span>'
            highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
        
        st.markdown(f"<div style='max-height: 300px; overflow-y: auto; border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem;'>{highlighted_text}</div>", unsafe_allow_html=True)

def display_semantic_search(document_text):
    """Display semantic search functionality"""
    st.markdown("<h3 class='feature-header'>🔍 Semantic Search</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Search within the document using natural language. This finds relevant sections even when exact keywords aren't present.
    </div>
    """, unsafe_allow_html=True)
    
    # Search input
    search_query = st.text_input("Search within document:", placeholder="e.g. termination conditions, payment terms")
    
    # Process search
    if search_query:
        with st.spinner("Searching document..."):
            search_results = semantic_search(search_query, top_k=5)
            
            if search_results:
                st.markdown("### Results")
                
                for i, result in enumerate(search_results):
                    # Display each result with context
                    sentence = result['sentence']
                    index = result['index']
                    
                    # Try to get some context (sentences before and after)
                    context = ""
                    if st.session_state.document_sentences:
                        start_idx = max(0, index - 1)
                        end_idx = min(len(st.session_state.document_sentences) - 1, index + 1)
                        
                        if start_idx < index:
                            context += f"<small>{st.session_state.document_sentences[start_idx]}</small> "
                        
                        context += f"<mark class='search-highlight'>{sentence}</mark>"
                        
                        if end_idx > index:
                            context += f" <small>{st.session_state.document_sentences[end_idx]}</small>"
                    else:
                        context = sentence
                    
                    st.markdown(f"""
                    <div class='search-result'>
                        <strong>Result {i+1}</strong> 
                        <div>{context}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No results found. Try a different search query.")

def display_document_qa(document_text):
    """Display the document Q&A section"""
    st.markdown("<h3 class='feature-header'>❓ Document Q&A</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Ask specific questions about the document and get AI-powered answers based on the content.
    </div>
    """, unsafe_allow_html=True)
    
    # Example questions based on common legal document queries
    st.markdown("**Example questions:**")
    examples = [
        "What are the main obligations described in this document?",
        "Is there a termination clause? What does it say?",
        "What happens if one party breaches the agreement?",
        "What is the payment schedule mentioned?",
        "Are there any warranties or representations in this document?"
    ]
    
    # Show examples as clickable buttons
    cols = st.columns(3)
    for i, example in enumerate(examples):
        col_index = i % 3
        if cols[col_index].button(f"Example {i+1}", key=f"example_{i}"):
            # Set the example as the question
            st.session_state.current_question = example
            st.rerun()
    
    # Input for question
    question = st.text_input("Ask a question about the document:", 
                            value=st.session_state.current_question)
    
    if st.button("Get Answer") and question:
        with st.spinner("Analyzing document to answer your question..."):
            answer = analyze_document_qa(document_text, question)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f"<div class='info-box'>{answer}</div>", unsafe_allow_html=True)
            
            # Show relevant sections from semantic search
            search_results = semantic_search(question, top_k=3)
            if search_results:
                with st.expander("View relevant document sections"):
                    for i, result in enumerate(search_results):
                        st.markdown(f"**Section {i+1}:** {result['sentence']}")
            
            # Update current question in session state
            st.session_state.current_question = question

def display_document_comparison(text1, text2):
    """Display the document comparison section"""
    st.markdown("<h3 class='feature-header'>🔄 Document Comparison</h3>", unsafe_allow_html=True)
    
    # Generate comparison if not in session state
    if not st.session_state.comparison_results:
        with st.spinner("Comparing documents..."):
            st.session_state.comparison_results = compare_documents(text1, text2)
    
    # Display comparison results
    st.markdown(f"<div class='info-box'>{st.session_state.comparison_results}</div>", unsafe_allow_html=True)
    
    # Add option to regenerate comparison
    if st.button("Regenerate Comparison"):
        with st.spinner("Regenerating comparison..."):
            st.session_state.comparison_results = compare_documents(text1, text2)
            st.rerun()
    
    # Add visual diff option
    st.markdown("---")
    st.markdown("<h4>Visual Difference View</h4>", unsafe_allow_html=True)
    
    if st.toggle("Show Visual Diff"):
        with st.spinner("Generating visual diff view..."):
            # Create simplified diff view (not using difflib's HTML formatter as it's too complex)
            text1_lines = text1[:15000].splitlines()  # Limit to prevent performance issues
            text2_lines = text2[:15000].splitlines()
            
            diff = difflib.unified_diff(text1_lines, text2_lines, lineterm='')
            
            # Process diff lines
            diff_html = []
            for line in list(diff)[2:]:  # Skip the first two metadata lines
                if line.startswith('+'):
                    diff_html.append(f"<div class='diff-added'>{line[1:]}</div>")
                elif line.startswith('-'):
                    diff_html.append(f"<div class='diff-removed'>{line[1:]}</div>")
                elif line.startswith('@@'):
                    diff_html.append(f"<div><strong>{line}</strong></div>")
                else:
                    diff_html.append(f"<div>{line}</div>")
            
            # Display diff with scrollable container
            st.markdown("<div style='max-height: 500px; overflow-y: auto; border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem;'>" + 
                      "".join(diff_html) + "</div>", unsafe_allow_html=True)
            
            st.caption("Green: Added in Document 2 | Red: Removed from Document 1")

# --- Main App Logic ---

# --- Main App Logic ---
def main():
    # Initialize session state
    init_session_state()
    
    # Setup sidebar
    uploaded_file, document_type, jurisdiction, summarization_level, comparison_file = sidebar_ui()
    
    # Display app header
    st.markdown("<h1 class='main-header'>LexScanCited Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Legal Document Analysis</p>", unsafe_allow_html=True)
    
    # Handle document upload and processing
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text content
            document_text = extract_text(uploaded_file)
            document_text = preprocess_text(document_text)
            
            # Calculate document hash to check for changes
            current_hash = hashlib.md5(document_text.encode()).hexdigest()
            
            # Check if this is a new document
            if current_hash != st.session_state.document_hash:
                # Reset document-specific session state
                st.session_state.document_content = document_text
                st.session_state.document_hash = current_hash
                st.session_state.document_summary = None
                st.session_state.embeddings_processed = False
                st.session_state.document_sentences = None
                st.session_state.sentence_embeddings = None
                st.session_state.faiss_index = None
                
                # Generate embeddings for search functionality
                with st.spinner("Creating document embeddings for semantic search..."):
                    sentences, embeddings, index = create_text_embeddings(document_text)
                    st.session_state.document_sentences = sentences
                    st.session_state.sentence_embeddings = embeddings
                    st.session_state.faiss_index = index
                    st.session_state.embeddings_processed = True
        
        # Handle comparison document if present
        if comparison_file is not None:
            with st.spinner("Processing comparison document..."):
                comparison_text = extract_text(comparison_file)
                comparison_text = preprocess_text(comparison_text)
                st.session_state.comparison_content = comparison_text
        
        # Create main tabs for different features
        tabs = st.tabs([
            "📄 Document Summary", 
            "📚 Legal Terms", 
            "🔍 Search", 
            "❓ Q&A", 
            "🔄 Compare Documents"
        ])
        
        # Tab 1: Document Summary
        with tabs[0]:
            display_document_summary(st.session_state.document_content, document_type, summarization_level)
            
            # Show document preview
            with st.expander("Document Preview", expanded=False):
                preview_text = st.session_state.document_content[:10000] + \
                    ("..." if len(st.session_state.document_content) > 10000 else "")
                st.text_area("Document Content", preview_text, height=300)
        
        # Tab 2: Legal Terms Lookup
        with tabs[1]:
            display_legal_terms_lookup(st.session_state.document_content)
        
        # Tab 3: Semantic Search
        with tabs[2]:
            if st.session_state.embeddings_processed:
                display_semantic_search(st.session_state.document_content)
            else:
                st.warning("Document embeddings not yet processed. Please wait a moment or refresh the page.")
        
        # Tab 4: Document Q&A
        with tabs[3]:
            display_document_qa(st.session_state.document_content)
        
        # Tab 5: Document Comparison
        with tabs[4]:
            if st.session_state.comparison_content is not None:
                # Display comparison between the two documents
                display_document_comparison(
                    st.session_state.document_content, 
                    st.session_state.comparison_content
                )
            else:
                st.info("Please upload a second document to compare with the current one.")
                st.markdown("""
                You can upload a second document using the file uploader in the sidebar under 'Document Comparison'.
                This feature is useful for comparing:
                - Different versions of a contract
                - Proposed changes to agreements
                - Variations in legal documents
                """)
    else:
        # No document uploaded yet
        st.markdown("""
        <div class='info-box'>
            <h3>Welcome to LexScanCite Pro! 👋</h3>
            <p>Upload a legal document (PDF or TXT) using the file uploader in the sidebar to get started.</p>
            <p>This AI-powered tool will help you analyze legal documents with features like:</p>
            <ul>
                <li>Automatic document summarization</li>
                <li>Legal terminology explanations</li>
                <li>Semantic search within documents</li>
                <li>Document Q&A capabilities</li>
                <li>Document comparison tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature preview section
        st.markdown("<h3 class='feature-header'>Feature Preview</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem;'>
                <h4>📄 Document Summary</h4>
                <p>Get AI-generated summaries of legal documents at different detail levels.</p>
                <ul>
                    <li>Understand document purpose quickly</li>
                    <li>Identify key rights and obligations</li>
                    <li>Spot important deadlines and conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;'>
                <h4>🔍 Semantic Search</h4>
                <p>Search for concepts, not just keywords, within your documents.</p>
                <ul>
                    <li>Find relevant sections using natural language</li>
                    <li>Locate information even when exact terms aren't used</li>
                    <li>Get context around search results</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem;'>
                <h4>📚 Legal Terms Lookup</h4>
                <p>Understand complex legal terminology with AI explanations.</p>
                <ul>
                    <li>Auto-extract important legal terms</li>
                    <li>Get simple explanations of complex concepts</li>
                    <li>Build your personal legal term database</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #E5E7EB; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;'>
                <h4>❓ Document Q&A</h4>
                <p>Ask questions about your document and get AI-powered answers.</p>
                <ul>
                    <li>Get answers to specific legal questions</li>
                    <li>Understand obligations and requirements</li>
                    <li>Find critical information quickly</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()