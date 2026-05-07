from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

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

# Initialize embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
    collection_name="sws_ai_docs"
)

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)

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
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
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
        response = llm.invoke(prompt)
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
