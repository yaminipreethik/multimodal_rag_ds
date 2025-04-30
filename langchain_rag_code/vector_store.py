import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

def create_vectorstore():
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings()
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever, id_key


def add_documents(retriever, id_key, texts, text_summaries, tables, table_summaries, images, image_summaries):
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    # table_ids = [str(uuid.uuid4()) for _ in tables]
    # summary_tables = [
    #     Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    # ]
    # retriever.vectorstore.add_documents(summary_tables)
    # retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add images
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
