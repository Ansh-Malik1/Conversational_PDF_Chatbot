from fastapi import APIRouter, UploadFile, File,FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

import os

load_dotenv()
import sqlite3
from datetime import datetime, timedelta
app = FastAPI()
os.environ["HUGGINGFACE_TOKEN"]=os.getenv("HUGGINGFACE_TOKEN")
class Query(BaseModel):
    question: str
@app.get("/")
def home():
    return {"message": "!"}

@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global documents
    documents = []
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_time TEXT,
            page_count INTEGER,
            chunk_count INTEGER
        )
    """)
    conn.commit()
    conn.close()
   
    if len(files) > 20:
        return JSONResponse(content={"error": "Max 20 files allowed."}, status_code=400)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()

    for file in files:
        contents = await file.read()
        temp_path = f"./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(contents)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        if len(docs) > 1000:
            return JSONResponse(content={"error": f"{file.filename} exceeds 1000 pages."}, status_code=400)

        splitter = RecursiveCharacterTextSplitter(chunk_size=5100, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        _ = FAISS.from_documents(chunks, embedding=embeddings)

        cursor.execute("INSERT INTO documents (filename, upload_time, page_count, chunk_count) VALUES (?, ?, ?, ?)",
                       (file.filename, datetime.now().isoformat(), len(docs), len(chunks)))
        documents.extend(docs)

    conn.commit()
    conn.close()
    
    return {"message": f"{len(files)} file(s) uploaded successfully."}


@app.post("/query")
async def ask_query(q: Query):

    session_store: dict[str, BaseChatMessageHistory] = {}
    llm = ChatGroq(api_key="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", model="Gemma2-9b-It")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   
    vectorStore=FAISS.from_documents(documents,embedding=embeddings)
    retriever=vectorStore.as_retriever()

    contextualize_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_prompt)


    system_prompt=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know.If any question is out of the pdf context," 
                "dont annswer it and say that it is not given in pdf."
                "Dont provide any extra information about a topic unless explicitly asked."
                "Keep the answer restricted to the text given in context"
                "\n\n"
                "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
    )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
    )

    response = conversational_rag_chain.invoke(
    {"input": q.question, "chat_history": []},
        config={"configurable": {"session_id": "abc"}}  # or any unique session string
)
    return {"answer": response["answer"]}

@app.get("/metadata")
async def get_all_metadata():
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents")
    rows = cursor.fetchall()
    conn.close()

    metadata = [
        {
            "id": row[0],
            "filename": row[1],
            "upload_time": row[2],
            "page_count": row[3],
            "chunk_count": row[4]
        }
        for row in rows
    ]

    return {"documents": metadata}