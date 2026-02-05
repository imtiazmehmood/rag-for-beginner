import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

 # Load documents from the specified directory
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(docs_path, glob='*.txt', loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")

    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents

def main():
    print("Starting document ingestion pipeline...")
    documents = load_documents()

if __name__ == "__main__":
    main()

