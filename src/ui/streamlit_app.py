import streamlit as st
import requests
import json
from pathlib import Path
import os
from typing import Dict, Any

# Constants
API_BASE_URL = "http://localhost:8000"
PREDEFINED_PDFS_DIR = Path("predefined-pdfs")

# Add these predefined questions
PREDEFINED_QUESTIONS = [
    "What happens when patients experiences characteristic epigastic abdominal pain and lipase >3x upper limit of normal?",
    "How much bolus should be repeated in case of urine output<0.5 or systemic BP < 90.",
    "What considerations should be taken for fontan repair lesions?",
    "How do I treat abdominal pain and kidney stone issues?",
    "Can I give NSAIDs for kidney stones and headaches?",
    "What patient education is needed for kidney stones during discharge?",
    "What are the clinical indications abdominal pain?",
    "What are the local complications for pancreatitis?",
    "What is the severity level for a pancreatitis bedside index of more than 3?"
]

def setup_page():
    st.set_page_config(
        page_title="Document Processing System",
        page_icon="📄",
        layout="wide"
    )
    st.title("Document Processing and QA System")

def upload_document(file) -> Dict[str, Any]:
    """Upload document to the API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def query_documents(query: str, collection_name: str = "default") -> Dict[str, Any]:
    """Query the document collection"""
    try:
        payload = {
            "query": query,
            "collection_name": collection_name
        }
        response = requests.post(f"{API_BASE_URL}/query/", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None

def get_collection_stats() -> Dict[str, Any]:
    """Get collection statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")
        return None

def main():
    setup_page()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page", ["Upload", "Query", "Statistics"])
    
    if page == "Upload":
        st.header("Document Upload")
        
        # Add predefined PDFs section
        st.subheader("Predefined Documents")
        predefined_files = list(PREDEFINED_PDFS_DIR.glob("*.pdf"))
        if predefined_files:
            selected_file = st.selectbox(
                "Select a predefined document",
                options=predefined_files,
                format_func=lambda x: x.name
            )
            
            if st.button("Process Predefined Document"):
                with st.spinner("Processing document..."):
                    # Check if document is already in cache
                    cache_file = Path("predefined-cache") / f"{selected_file.stem}_results.json"
                    if cache_file.exists():
                        st.info("Using cached version of the document")
                        # Load cached results
                        with open(cache_file, 'r') as f:
                            cached_data = json.loads(f.read())
                            st.success("Document loaded from cache successfully!")
                            st.json(cached_data)
                    else:
                        # Only process if not cached
                        with selected_file.open("rb") as file:
                            result = upload_document(file)
                            if result:
                                st.success("Document processed successfully!")
                                st.json(result)
        
        # Existing upload functionality
        st.subheader("Custom Document Upload")
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt", "doc", "docx"])
        
        if uploaded_file:
            if st.button("Process Custom Document"):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success("Document processed successfully!")
                        st.json(result)
    
    elif page == "Query":
        st.header("Document Query")
        
        # Add predefined questions section
        query_type = st.radio("Query Type", ["Predefined Questions", "Custom Question"])
        
        if query_type == "Predefined Questions":
            query = st.selectbox("Select a question:", PREDEFINED_QUESTIONS)
        else:
            query = st.text_input("Enter your question:")
        
        if query:
            if st.button("Submit Query"):
                with st.spinner("Processing query..."):
                    result = query_documents(query)
                    if result:
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        st.subheader("Sources")
                        for source in result["sources"]:
                            st.markdown(f"- {source}")
                        
                        st.subheader("Confidence")
                        st.progress(result["confidence"])
                        
                        if "reasoning_chain" in result:
                            st.subheader("Reasoning Chain")
                            for step in result["reasoning_chain"]:
                                st.markdown(f"- {step}")
    
    else:  # Statistics page
        st.header("Collection Statistics")
        stats = get_collection_stats()
        
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats["total_documents"])
            
            with col2:
                st.metric("Collections", len(stats["collections"]))
            
            with col3:
                st.metric("Embedding Dimensions", stats["embedding_dimensions"])
            
            st.subheader("Available Collections")
            for collection in stats["collections"]:
                st.markdown(f"- {collection}")

if __name__ == "__main__":
    main() 