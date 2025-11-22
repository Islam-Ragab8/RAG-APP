# 📘 RAG Chat with PDFs

This project allows you to **chat with multiple PDF documents** using a Retrieval-Augmented Generation (RAG) system.  
It extracts text from PDFs, creates embeddings, stores them in a vector database, and answers user questions using a remote LLM.

---

## 🛠 Features

- Upload multiple PDF files at once.
- Automatically extract text from PDFs.
- Split text into manageable chunks for better retrieval.
- Convert text chunks into embeddings using `sentence-transformers`.
- Store embeddings in a **FAISS vector store** for fast retrieval.
- Ask questions and get precise answers from your PDFs.
- Extract only the **Helpful Answer** for clean, concise responses.

---

## 📦 Requirements

- Python 3.9.4
- Install dependencies:

```bash
pip install -r requirements.txt
```
## 📝 Project Structure
```bash
project/
│
├─ modules/
│   ├─ RAG.py          # Core RAG pipeline: PDF reading, chunking, vectorstore, query
│   └─ Request.py      # RemoteLLM class to call external LLM API
│
├─ app.py              # Main Streamlit app
├─ requirements.txt    # Python dependencies
└─ README.md           # Project documentation
```

## ⚙ How It Works
---
1. Upload PDFs: User uploads one or more PDF files.

2. Extract Text: PyPDF2 reads all pages from PDFs.

3. Chunk Text: Text is split into chunks using CharacterTextSplitter.

4. Create Embeddings: Each chunk is converted to embeddings using sentence-transformers/all-MiniLM-L6-v2.

5. Store in Vector DB: Chunks and embeddings are stored in FAISS for fast similarity search.

6. Ask Questions: User asks a question in the chat.

7. Retrieve + Generate: System retrieves relevant chunks and sends them to a Mistral LLM (via remote API) for answer generation.

8. Show Answer: Only the text after "Helpful Answer:" is displayed for clarity.

---

## ⚡ Usage

1-Clone the repository:
```bash
git clone https://github.com/Islam-Ragab8/RAG-APP

cd  RAG-APP
```
2-Create a .env file for environment variables (optional, e.g., API keys):
```bash
 API_KEY=your_api_key_here
 ```

3-Run the Streamlit app:
 ```bash
 streamlit run app.py
 ```



