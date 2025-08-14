from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ===== Document Loading & Splitting =====
print("Loading document...")
loader = TextLoader("my_document.txt")
documents = loader.load()

print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# ===== Create Vector Store =====
print("Creating vector store...")
embedding_model = OllamaEmbeddings(model="qwen2.5")
vector_store = Chroma.from_documents(chunks, embedding_model)

# ===== LLM Setup =====
print("Setting up LLM...")
llm = OllamaLLM(model="qwen2.5")

# ===== Memory for Conversation =====
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ===== Conversational Retrieval Chain =====
print("Setting up Conversational Retrieval Chain...")
retriever = vector_store.as_retriever()
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ===== Request Model =====
class ChatRequest(BaseModel):
    message: str

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Endpoint =====
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    query = chat_request.message

    # If the message is "code 3000", respond but don't save in memory
    if query.strip().lower() == "code 3000":
        return {
            "response": "Hello there! I'm Nova, your friendly AI buddy here to chat and help. How are you today? Could you share your name with me and maybe some fun facts about yourself that others might find interesting? 😊"
        }

    result = qa_chain.invoke({"question": query})
    return {"response": result["answer"]}
