import os

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from modules.Request import RemoteLLM

load_dotenv()
api_url=os.getenv("API_URL")
api_key=os.getenv("SEKRT_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={"device": "cpu"}
    )                                   
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def conversation_chain(vectorstore):
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm=RemoteLLM(
    api_url="https://f391b9dfa7d7.ngrok-free.app/generate",
    api_key=api_key
    ) 

    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

    
def query_user(question):
    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    with st.chat_message("user"):
        st.write(question)

    response = st.session_state.conversation({"question": question})

    answer_text = response.get("answer", "")
    if "Helpful Answer:" in answer_text:
        answer_text = answer_text.split("Helpful Answer:")[-1].strip()

    with st.chat_message("assistant"):
        st.write(answer_text)
    st.session_state.chat_history.append((question, answer_text))
    return answer_text
