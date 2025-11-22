import streamlit as st
from modules.RAG import get_pdf_text, get_text_chunks, get_vectorstore, conversation_chain, query_user
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chat", page_icon="📘")

    st.title(" Chat with Multiple PDFS  :books:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask anything from your PDFs...")

    if user_input:
        query_user(user_input)

        
    with st.sidebar:
        st.header("📤 Upload PDFs")

        pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Reading files..."):
                raw_text = get_pdf_text(pdf_files)
                chunks = get_text_chunks(raw_text)
                vectordb = get_vectorstore(chunks)
                st.session_state.conversation = conversation_chain(vectordb)
            st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()
