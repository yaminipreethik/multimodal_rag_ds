import gradio as gr
from huggingface_hub import InferenceClient
import time
import requests
from PIL import Image
import io
import base64
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Initialize the Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_API_KEY"),  # Set in Hugging Face Spaces secrets
)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Pinecone index configuration
index_name = "gradio-rag-index"
dimension = 768  # Match SentenceTransformer dimension

# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
uploaded_pdf_names = set()  # Track uploaded PDF names

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            split_point = max(last_period, last_newline)
            
            if split_point > start:
                end = split_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    
    return chunks

def store_pdf(pdf_path, pdf_name):
    """Process PDF and store in Pinecone"""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append((f"{pdf_name}-{i}", embedding, {
            "pdf_name": pdf_name,
            "chunk_id": i,
            "text": chunk
        }))
    
    # Upsert in batches for free tier limits
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
    
    uploaded_pdf_names.add(pdf_name)
    return f"Processed and stored '{pdf_name}' with {len(chunks)} chunks"

def retrieve_context(query, top_k=5):
    """Retrieve relevant context from Pinecone"""
    query_embedding = embedding_model.encode(query).tolist()
    
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return "\n\n".join(
        [match.metadata["text"] for match in response.matches]
    )

def process_ticket(text, image=None, pdf=None):
    try:
        # Handle PDF upload
        pdf_result = ""
        if pdf is not None:
            pdf_name = os.path.basename(pdf)
            pdf_result = store_pdf(pdf, pdf_name)
        
        # Process image if provided
        if image is not None:
            img = Image.open(image)
            max_size = 800
            ratio = min(max_size/img.width, max_size/img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='png', quality=85)
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]
                }
            ]
        # else:
        if "?" in text and len(uploaded_pdf_names) > 0:
            context = retrieve_context(text)
            rag_prompt = f"""Answer based on context:
            
Question: {text}... Context:
{context}

Provide a clear answer using only the context. If irrelevant, state so.
            """
            messages = [{"role": "user", "content": rag_prompt}]
        else:
            messages = [{"role": "user", "content": text}]

        # API call with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct",  
                    messages=messages
                )
                response = completion.choices[0].message.content
                if pdf_result:
                    response = f"{pdf_result}\n\n{response}"
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                raise
        
        return "Failed to process request after retries."

    except Exception as e:
        print(f"Error processing ticket: {e}")
        return f"An error occurred: {str(e)}"

def create_interface():
    with gr.Blocks(title="Multimodal AI Assistant with PDF RAG") as demo:
        gr.Markdown("# Multimodal AI Assistant with PDF RAG")
        
        with gr.Tab("Chat & Upload"):
            text_input = gr.Textbox(
                label="Message or Question",
                placeholder="Describe an issue or ask about your PDFs",
                lines=4)
            
            with gr.Row():
                image_input = gr.Image(label="Upload an image (Optional)", type="filepath")
                pdf_input = gr.File(label="Upload a PDF (Optional)", file_types=[".pdf"])
            
            submit_btn = gr.Button("Submit")
            output = gr.Textbox(label="Response", lines=10)
            
            submit_btn.click(
                fn=process_ticket,
                inputs=[text_input, image_input, pdf_input],
                outputs=output)
        
        with gr.Tab("PDF Database"):
            gr.Markdown("### Uploaded PDFs")
            
            def list_pdfs():
                if not uploaded_pdf_names:
                    return "No PDFs uploaded yet."
                return "\n".join(f"- {pdf}" for pdf in uploaded_pdf_names)
            
            pdf_list = gr.Textbox(label="Uploaded PDFs", lines=5)
            refresh_btn = gr.Button("Refresh List")
            refresh_btn.click(fn=list_pdfs, inputs=[], outputs=pdf_list)
    
    demo.launch(debug=True)

create_interface()
