# 📄 Chat with PDF RAG Bot  

A simple and efficient **Retrieval-Augmented Generation (RAG)** bot that allows users to **upload PDFs** and ask questions about their content. The system extracts text, generates embeddings, retrieves relevant information, and provides AI-generated responses using **Gemini 1.5 Flash**.  

## 🚀 Features  
✅ **Upload PDFs** and extract text using PyPDF2.  
✅ **Generate embeddings** with `all-MiniLM-L6-v2`.  
✅ **Fast retrieval** using FAISS.  
✅ **AI-powered answers** generated by Gemini 1.5 Flash.  
✅ **Simple and efficient** without requiring a complex vector database.  
✅ **Supports ChromaDB and Milvus** for scalable vector storage (optional).  

## 🛠 Tech Stack  
- **UI**: Streamlit  
- **Embedding Model**: all-MiniLM-L6-v2  
- **Vector Store**: FAISS (default), ChromaDB, Milvus  
- **Text Extraction**: PyPDF2  
- **RAG Framework**: LangChain  
- **LLM**: Gemini 1.5 Flash  

## 📝 Why FAISS Instead of a Vector Database?  
I have worked with **ChromaDB** and **Milvus** for vector storage, but for this use case, **FAISS** was the preferred choice due to:  
- ✅ **Simplicity**: No external database setup required.  
- ✅ **Efficiency**: In-memory search provides **fast and accurate** retrieval.  
- ✅ **Ease of Integration**: Works seamlessly with LangChain.  

For larger applications, **ChromaDB and Milvus** can be integrated when persistence and scalability are needed.  

## 🔧 Installation  
First, install the required dependencies:  
```bash
pip install streamlit PyPDF2 langchain faiss-cpu sentence-transformers google-generativeai chromadb pymilvus
