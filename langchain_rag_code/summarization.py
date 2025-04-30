from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def summarize_texts(texts):
    prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.
Table or text chunk: {element}
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    return text_summaries

def summarize_tables(tables):
    prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.
Table or text chunk: {element}
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    return table_summaries

def summarize_images(images):
    """
    Summarize each image using GPT-4o or similar multimodal model.
    Each image is a base64-encoded string.
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    summaries = []
    for image_b64 in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in detail. For context, the image is part of a research paper explaining the transformers architecture. Be specific about graphs, such as bar plots."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]
        summary = llm.invoke(messages)
        summaries.append(summary.content if hasattr(summary, "content") else summary)
    return summaries
