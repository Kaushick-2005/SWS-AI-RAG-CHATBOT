from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
embedding_model = None
vector_store = None
llm = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=get_embedding_model(),
            collection_name="sws_ai_docs"
        )
    return vector_store

def get_llm():
    global llm
    if llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0
        )
    return llm

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def read_root():
    return {"message": "SWS AI RAG Chatbot API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Retrieve relevant chunks
        vs = get_vector_store()
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(request.question)
        
        if not docs:
            return ChatResponse(
                answer="I don't have that information in the company documents.",
                sources=[]
            )
        
        # Extract sources
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a helpful HR assistant for SWS AI. Answer the user's question based ONLY on the provided company documents. 
If the answer is not in the documents, respond: "I don't have that information in the company documents."

Company Documents:
{context}

User Question: {request.question}

Answer:"""
        
        # Get response from LLM
        llm_instance = get_llm()
        response = llm_instance.invoke(prompt)
        answer = response.content
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
