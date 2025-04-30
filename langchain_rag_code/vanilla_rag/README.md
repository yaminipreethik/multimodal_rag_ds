# Multimodal-RAG

Multimodal-RAG is an AI-powered assistant that combines text, image, and PDF processing capabilities. It leverages Pinecone for document storage and retrieval, Hugging Face for inference, and Gradio for an interactive user interface.

## Features

- **Chat Interface**: Ask questions or describe issues, and get AI-powered responses.
- **PDF Upload and Retrieval**: Upload PDFs, extract text, and retrieve relevant context for queries using Pinecone.
- **Image Processing**: Upload images to include in your queries.
- **Contextual Responses**: Combines user queries with relevant PDF context for accurate answers.

## Tech Stack

- **Gradio**: Interactive UI for user input and output.
- **Hugging Face**: Inference client for AI model interactions.
- **Pinecone**: Vector database for storing and retrieving document embeddings.
- **Sentence Transformers**: For generating embeddings from text.
- **PyPDF2**: For extracting text from PDFs.
- **Python**: Core programming language.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multimodel-rag

2. Install Dependencies:
    ```bash
    pip install -r requirements.txt

3. Setup Env variables:
    ```bash
    HF_API_KEY=your_huggingface_api_key
    PINECONE_API_KEY=your_pinecone_api_key

4. Run application with gradio
    ```bash
    python src/app.py # or
    gradio src/app.py