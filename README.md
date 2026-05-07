# SWS AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python that answers employee questions about SWS AI company policies using internal PDF documents.

## 🎯 Features

- **RAG-Powered Search**: Retrieves relevant information from company documents
- **Natural Language Q&A**: Ask questions in plain English
- **Source Attribution**: Shows which documents were used to answer your question
- **No Hallucination**: Only answers based on provided company documents
- **Beautiful Chat UI**: Modern, responsive web interface
- **Fast & Accurate**: Uses Chroma vector database and Claude AI

## 📋 Project Structure

```
sws-ai-rag-chatbot/
├── backend/
│   ├── app.py                 # FastAPI backend with RAG pipeline
│   ├── ingest.py              # Document ingestion and embedding script
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variables template
│   ├── documents/              # Company PDF documents (10 files)
│   └── chroma_db/              # Vector database storage
├── frontend/
│   └── index.html              # React-free chat UI (HTML/CSS/JS)
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API Key (for Claude)

### 1. Clone the Repository

```bash
git clone https://github.com/Kaushick-2005/SWS-AI-RAG-CHATBOT.git
cd SWS-AI-RAG-CHATBOT
```

### 2. Set Up Environment Variables

```bash
cd backend
cp .env.example .env
# Edit .env and add your Anthropic API Key
```

### 3. Install Dependencies

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Ingest Documents

Run the ingestion script to process PDFs and create embeddings:

```bash
python ingest.py
```

This will:
- Load all 10 company PDF documents
- Extract and chunk text (500 chars per chunk, 50 char overlap)
- Generate embeddings using HuggingFace sentence-transformers
- Store in local Chroma vector database

### 5. Start the Backend API

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### 6. Open the Chat UI

Open `frontend/index.html` in your web browser:
```bash
# On Windows
start frontend/index.html

# On macOS
open frontend/index.html

# On Linux
xdg-open frontend/index.html
```

Or use a local server:
```bash
# Python 3.7+
python -m http.server 8080 --directory frontend
```

Then visit `http://localhost:8080`

## 🏗️ Architecture

### RAG Pipeline

1. **Document Ingestion** (`ingest.py`)
   - Loads PDFs using PyMuPDF
   - Chunks text with RecursiveCharacterTextSplitter (500 char chunks, 50 char overlap)
   - Generates embeddings with HuggingFace all-MiniLM-L6-v2

2. **Vector Database** (Chroma)
   - Stores embeddings with metadata (source, page number)
   - Local persistence for offline access
   - Fast semantic search

3. **Retrieval** (app.py)
   - Embeds user question with same model
   - Retrieves top-5 most relevant chunks
   - Includes source document names

4. **Generation** (Claude 3.5 Sonnet)
   - Receives question + retrieved context
   - Generates grounded answer
   - System prompt ensures no hallucination

### Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Vector DB** | Chroma | Local, easy setup, perfect for prototypes |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Free, efficient, no API key needed |
| **LLM** | Claude 3.5 Sonnet | State-of-the-art reasoning, reliable |
| **Backend** | FastAPI | Modern, fast, automatic API docs |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, instant deployment |
| **PDF Processing** | PyMuPDF + pdfplumber | Reliable text extraction |

## 📊 Configuration

### Chunking Strategy

- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Why**: Balances context preservation with retrieval efficiency

### Retrieval

- **Top-K**: 5 chunks
- **Distance Metric**: Cosine similarity (Chroma default)
- **Why**: Ensures sufficient context while avoiding noise

### LLM Settings

- **Temperature**: 0 (deterministic, factual answers)
- **System Prompt**: Instructs model to only use provided documents
- **Fallback**: "I don't have that information in the company documents."

## 💬 Sample Queries

Test your chatbot with these questions:

- "What is the annual leave policy at SWS AI?"
- "How many days of sick leave do employees get?"
- "What is the notice period for resignation?"
- "What tools does SWS AI use for communication?"
- "What is the password policy for company systems?"
- "How are performance reviews conducted?"
- "What are the WFH guidelines?"
- "Does SWS AI offer health insurance?"

## 🔧 API Endpoints

### `/api/chat` (POST)

**Request:**
```json
{
  "question": "What is the leave policy?"
}
```

**Response:**
```json
{
  "answer": "SWS AI provides...",
  "sources": ["SWS-AI-leave-policy.pdf", "SWS-AI-hr-policy.pdf"]
}
```

### `/` (GET)

Health check endpoint - returns API status.

## 📝 API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🛠️ Development

### Adding New Documents

1. Place PDF files in `backend/documents/`
2. Run `python ingest.py` to re-index
3. Restart the backend API

### Modifying Chunking Strategy

Edit `backend/ingest.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for larger chunks
    chunk_overlap=50     # Increase for more overlap
)
```

### Changing the LLM

Edit `backend/app.py`:
```python
from langchain_openai import ChatOpenAI  # or other providers
llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
```

## 📦 Alternative Setups

### Using OpenAI instead of Anthropic

1. Install: `pip install langchain-openai`
2. Update `app.py`:
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
   ```

### Using Pinecone Vector DB

1. Create Pinecone account & index
2. Install: `pip install pinecone-client`
3. Update `app.py` and `ingest.py` to use Pinecone

### Using Local Ollama

1. Install Ollama from ollama.ai
2. Update `app.py` to use `ChatOllama`

## 🐛 Troubleshooting

### CORS Issues

If frontend can't reach backend, ensure CORS middleware is enabled in `app.py` (already included).

### PDF Not Found

Ensure PDFs are in `backend/documents/` before running `ingest.py`.

### "No module named..." Error

Make sure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Slow Responses

- Increase `top-k` in retrieval (more context = slower)
- Use a faster LLM model
- Batch indexing for large datasets

## 📄 License

All company documents are confidential and proprietary to SWS AI.

## 👤 Author

Kaushick-2005

## 🤝 Contributing

Contributions welcome! Please ensure:
- Clean commit history (commit every 15 minutes during development)
- Descriptive commit messages
- Updated README for significant changes

## 📞 Support

For issues or questions, create a GitHub issue in the repository.

---

**Made with ❤️ for SWS AI**
