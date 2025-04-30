def retrieve_docs(retriever, query):
    docs = retriever.invoke(query)
    return docs
