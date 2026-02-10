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


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    print(f"\nSplitting documents into chunks of {chunk_size} characters with {chunk_overlap} overlap...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"Creating vector store in directory {persist_directory}...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created in directory {persist_directory}")
    return vector_store

def main():
    print("Starting document ingestion pipeline...")
    #1. Lofinng documents
    documents = load_documents()
    #2. Splitting documents into chunks
    chunks = split_documents(documents)
    #3. Creating vector store from chunks
    vectorstore = create_vector_store(chunks)
    print("Document ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main()

