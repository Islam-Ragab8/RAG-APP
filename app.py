import streamlit as st
from modules.RAG import get_pdf_text
from modules.RAG import get_chunk_text

st.header("Chat with multiple pdf")
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])