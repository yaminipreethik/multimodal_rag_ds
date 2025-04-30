# setup_dependencies.py

import subprocess
import sys

def install_python_packages():
    packages = [
        "unstructured[all-docs]", "pillow", "lxml", "chromadb", "tiktoken",
        "langchain", "langchain-community", "langchain-openai", "langchain-groq",
        "python_dotenv"
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-Uq", pkg])

if __name__ == "__main__":
    install_python_packages()
