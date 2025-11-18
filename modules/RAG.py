from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2



def get_pdf_text(file):
    text=""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() 
    return text