
# 🚀 Conversational PDF Chatbot

A full-stack, Dockerized **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions about their contents through a conversational chatbot interface.
Demo Video : https://www.loom.com/share/c2143b0b08534a67b74036b8f54c5630?sid=0875e334-0f13-44ec-a1e8-e2d98931ba87
Built with:
- 🧠 LangChain (RAG pipeline)
- 🤖 HuggingFace Embeddings
- 🔍 FAISS Vector Store
- 🗂️ Streamlit (frontend)
- ⚙️ FastAPI (backend API)
- 🐳 Docker + Docker Compose

---

## 📌 Features

- Upload up to **20 PDFs**, each with **1000-page max**
- Extract, chunk, and embed documents for vector search
- Query via chatbot or REST API
- Document metadata storage using SQLite
- Dockerized for easy local or cloud deployment
- Modular architecture (Streamlit + FastAPI)

---

## 🗂️ Project Structure

```

.
├── app.py                  # Streamlit chatbot UI
├── backend/
│   ├── main.py             # FastAPI entrypoint
├── Dockerfile              # Dockerfile for Streamlit
├── backend/Dockerfile      # Dockerfile for FastAPI
├── docker-compose.yml      # Compose setup for both services
├── requirements.txt
├── .env.example            # Example environment config
├── metadata.db             # SQLite metadata (auto-created)
├── README.md

````

---

## ⚙️ Setup

### 🐳 1. Docker (Recommended)

```bash
# Clone project
git clone https://github.com/your-username/PanScience-Assignment.git
cd PanScience-Assignment

# Copy and edit .env
cp .env.example .env

# Build & run
docker compose up --build
````

### 🔌 2. Local (Manual)

```bash
pip install -r requirements.txt
python app.py  # for Streamlit UI
uvicorn backend.main:app --reload  # in another terminal for FastAPI
```

---

## 📄 Environment Variables

Create a `.env` file:

```
HUGGINGFACE_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

---

## 💬 Streamlit Chat UI

Accessible at:
👉 [http://localhost:8501](http://localhost:8501)

* Upload PDFs
* Ask questions
* Session-aware chat
* PDF-only context enforcement

---

## 🌐 FastAPI Endpoints

Accessible at:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

| Endpoint     | Method | Description                     |
| ------------ | ------ | ------------------------------- |
| `/upload/`   | POST   | Upload one or more PDF files    |
| `/query/`    | POST   | Ask a question about documents  |
| `/metadata/` | GET    | View document metadata (SQLite) |

Example query:

```json
POST /query/
{
  "question": "What is the summary of document 1?"
}

## 🧱 Deployment

### 📍 Render / Railway / VM

* Push this repo to GitHub or SCP to a VM
* Set `.env` environment variables
* Run:

```bash
docker compose up --build -d
```

Then access:

* Streamlit: [http://your-ip:8501](http://your-ip:8501)
* FastAPI: [http://your-ip:8000/docs](http://your-ip:8000/docs)

---

## 🧠 Technologies Used

| Stack       | Purpose                  |
| ----------- | ------------------------ |
| LangChain   | RAG pipeline & chains    |
| HuggingFace | Text embeddings          |
| FAISS       | Vector similarity search |
| PyPDFLoader | PDF parsing              |
| FastAPI     | REST API backend         |
| Streamlit   | Chat UI frontend         |
| SQLite      | Document metadata        |
| Docker      | Containerization         |

---

## ✅ Deliverables (Checklist)

* [x] RAG pipeline with document upload
* [x] Conversational chatbot interface
* [x] REST API with `/upload`, `/query`, `/metadata`
* [x] Document metadata in SQLite
* [x] Docker Compose setup
* [x] `.env.example` + README
* [x] Public deployment (optional)

---

## 🙌 Author

**Ansh Malik**
LLM Specialist | RAG | LangChain | Docker
GitHub: [Ansh-Malik1](https://github.com/Ansh-Malik1)

---

## 📬 Contact & Submission

* Clone or download this repo
* Or access deployed version (if applicable)
* Contact for deployment help or further demo
