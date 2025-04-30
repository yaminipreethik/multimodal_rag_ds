# Multi-modal RAG with LangChain

This repository provides a modular Python implementation of a **Multi-modal Retrieval-Augmented Generation (RAG) pipeline** using [LangChain](https://github.com/langchain-ai/langchain), supporting both text and image data from PDF documents. The pipeline extracts, summarizes, indexes, and enables question-answering over text, tables, and images using state-of-the-art LLMs and multimodal models.

---

## Features

- **PDF Extraction**: Extracts texts, tables, and images from PDFs using [unstructured](https://github.com/Unstructured-IO/unstructured).
- **Summarization**: Summarizes text, tables, and images using LLMs (Groq, OpenAI GPT-4o).
- **Vector Indexing**: Stores summaries in a vector database (ChromaDB) for efficient retrieval.
- **Multi-modal Retrieval**: Supports retrieval and RAG over both text and images.
- **RAG Pipeline**: Answers user queries by retrieving relevant context (text, tables, images) and generating answers with LLMs.

---

## Project Structure

```
multimodal_rag
├── config.py # API key and environment setup
├── pdf_extraction.py # PDF partitioning and element separation
├── summarization.py # Summarization for text, tables, images
├── vectorstore.py # Vector store and retriever setup
├── retrieval.py # Simple retrieval utilities
├── rag_chain.py # RAG pipeline construction
├── utils.py # Utility functions (e.g., display images)
├── main.py # Pipeline orchestration and example usage
```

---

## Setup

### 1. System Dependencies

Install required system packages:

**Linux:**


sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev


### 2. Python Dependencies

Install Python packages:
```
pip install -U "unstructured[all-docs]" pillow lxml chromadb tiktoken
langchain langchain-community langchain-openai langchain-groq python_dotenv
```

### 3. API Keys

Set your API keys in `config.py`:
```
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_KEY"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_KEY"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

---

## Usage

1. Place your PDF file (e.g., `attention.pdf`) in the project directory.
2. Run the main script:
```
python main.py
```
3. The script will invoke a gradio app, where you will get a link to a UI. In the gradio app, one can upload the pdf and ask any question to receive the answer.

---

## Pipeline Overview

1. **PDF Extraction**:  
Extracts text, tables, and images using `unstructured.partition.pdf.partition_pdf`.

2. **Element Separation**:  
Separates extracted elements into text, tables, and images.

3. **Summarization**:  
- Text and tables are summarized with Groq's Llama-3.1-8b-instant.
- Images are summarized with OpenAI's GPT-4o (multimodal).

4. **Vector Store Creation**:  
Summaries are embedded and stored in ChromaDB; original elements are linked for full retrieval.

5. **Retrieval and RAG**:  
- User queries are embedded and used to retrieve relevant summaries.
- The RAG chain builds a prompt with retrieved text and images, then generates an answer using GPT-4o.

---

## References

- [LangChain Inspiration](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev)
- [Multivector Storage](https://python.langchain.com/docs/how_to/multi_vector/)

