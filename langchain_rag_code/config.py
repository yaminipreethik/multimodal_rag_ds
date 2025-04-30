import os

def set_env():
    os.environ["OPENAI_API_KEY"] = "OPEN API KEY"
    os.environ["GROQ_API_KEY"] = "GROQ API KEY"
    os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN API KEY"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"