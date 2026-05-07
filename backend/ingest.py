import os
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_PATH = "documents"

documents = []

# READ PDFs
for file in os.listdir(DOCS_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DOCS_PATH, file)

        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):

            page = pdf[page_num]

            text = page.get_text()

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "page": page_num + 1
                    }
                )
            )

print(f"Loaded {len(documents)} pages")


# SPLIT INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STORE IN CHROMA DB
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

vectordb.persist()

print("Embeddings stored successfully in ChromaDB")