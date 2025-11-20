import streamlit as st
from modules.RAG import get_pdf_text
from modules.RAG import get_text_chunks
from modules.RAG import get_vectorstore
from modules.RAG import conversation_chain
from modules.RAG import query_user
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    st.header("Chat with multiple pdf")

    user_question=st.text_input("Ask Question about your pdf")
    if user_question:
        query_user(user_question)

    with st.sidebar:
        st.subheader("Upload your pdf file")
        pdf_file = st.file_uploader("Upload your pdf", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_file)

                text_chunks=get_text_chunks(raw_text)

                vectordb=get_vectorstore(text_chunks)

                st.session_state.conversation=conversation_chain(vectordb)

if __name__ == "__main__":
    main()