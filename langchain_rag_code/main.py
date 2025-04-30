from config import set_env
from pdf_extraction import extract_pdf_elements, separate_elements, get_images_base64
from summarization import summarize_texts, summarize_tables, summarize_images
from vector_store import create_vectorstore, add_documents
from retrieval import retrieve_docs
from rag_chain import get_rag_chain, get_chain_with_sources
from utils import display_base64_image

def main():
    set_env()
    file_path = "attention.pdf"
    chunks = extract_pdf_elements(file_path)
    tables, texts = separate_elements(chunks)
    images = get_images_base64(chunks)

    text_summaries = summarize_texts(texts)
    table_summaries = summarize_tables(tables)
    image_summaries = summarize_images(images)

    retriever, id_key = create_vectorstore()
    add_documents(retriever, id_key, texts, text_summaries, tables, table_summaries, images, image_summaries)

    # Example retrieval
    docs = retrieve_docs(retriever, "Who are the authors of the paper?")
    for doc in docs:
        print(str(doc) + "\n\n" + "-" * 80)

    # RAG pipeline
    chain = get_rag_chain(retriever)
    response = chain.invoke("What is the attention mechanism?")
    print("RAG Response:", response)

    chain_with_sources = get_chain_with_sources(retriever)
    response = chain_with_sources.invoke("What is multihead?")
    print("Response:", response['response'])
    print("\n\nContext:")
    for text in response['context']['texts']:
        print(text.text)
        print("Page number: ", text.metadata.page_number)
        print("\n" + "-"*50 + "\n")
    for image in response['context']['images']:
        display_base64_image(image)

if __name__ == "__main__":
    main()
