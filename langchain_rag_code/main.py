import gradio as gr

from config import set_env
from pdf_extraction import extract_pdf_elements, separate_elements, get_images_base64
from summarization import summarize_texts, summarize_tables, summarize_images
from vector_store import create_vectorstore, add_documents
from rag_chain import get_rag_chain

set_env()  # Ensure API keys are set

def run_pipeline(pdf_file, question):
    # pdf_file is a string file path provided by Gradio
    chunks = extract_pdf_elements(pdf_file)
    tables, texts = separate_elements(chunks)
    images = get_images_base64(chunks)

    text_summaries = summarize_texts(texts)
    table_summaries = summarize_tables(tables)
    image_summaries = summarize_images(images)

    retriever, id_key = create_vectorstore()
    add_documents(retriever, id_key, texts, text_summaries, tables, table_summaries, images, image_summaries)

    chain = get_rag_chain(retriever)
    response = chain.invoke(question)

    return response

with gr.Blocks() as demo:
    gr.Markdown("# Multi-modal RAG with LangChain - Gradio UI")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        question_input = gr.Textbox(label="Enter your question", placeholder="Ask something about the document...", lines=2)
    output_text = gr.Textbox(label="Answer", lines=10)

    def on_submit(pdf_file, question):
        if pdf_file is None:
            return "Please upload a PDF file."
        if not question:
            return "Please enter a question."
        return run_pipeline(pdf_file, question)

    submit_btn = gr.Button("Get Answer")
    submit_btn.click(on_submit, inputs=[pdf_input, question_input], outputs=output_text)

if __name__ == "__main__":
    demo.launch(share=True)
