import uvicorn
import os
import uuid
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from supabase_tools import get_baby_report, get_chat_history, add_to_chat_history, get_baby_id_for_user
from fastapi.middleware.cors import CORSMiddleware



# Load environment variables from .env file
load_dotenv()
# Initialize FastAPI app
app = FastAPI(
    title="Baby Care Chatbot API",
    description="A secure API for a baby care chatbot that remembers conversation history.",
    version="4.0.0",
)
origins = os.getenv("CORS_ORIGINS", "http://localhost:8081,https://babyai-production.up.railway.app").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tüm domainlere izin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- Security ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is not used, but required

def get_current_user_token(token: str = Depends(oauth2_scheme)) -> str:
    """Dependency to get the bearer token from the request."""
    return token

# --- Pydantic Models ---
class ChatInput(BaseModel):
    question: str
    # baby_id is no longer needed from the client
    report_type: Optional[str] = "end_of_day_summary"
    session_id: Optional[str] = None

class ChatOutput(BaseModel):
    response: str
    session_id: str = Field(description="The unique identifier for the conversation session.")

# --- Prompt Templates ---
def get_context_prompt():
    """Returns prompt for when baby context is available."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are BabyAI, a helpful and caring baby care assistant.
                You will be given a detailed report containing all available information about a specific baby.
                Your task is to answer the user's question based ONLY on the information provided in that report and the previous conversation history.
                Do not make up information. If the answer is not in the report, politely say that you do not have that specific information.
                Your answers must be in Turkish.
                Here is the baby's report:

                {report_text}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

def get_general_prompt():
    """Returns prompt for general baby care questions without specific context."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are BabyAI, a helpful and caring baby care assistant.
                You provide general information about baby care, development, and common questions that parents might have.
                You have general knowledge about baby care but do not have access to specific information about a particular baby.
                When answering:
                - Provide helpful, evidence-based general advice
                - Always mention that this is general information and parents should consult with healthcare professionals
                - Recommend consulting pediatricians for specific medical concerns
                - Be encouraging and supportive in your responses
                - If the question requires specific medical advice, direct them to consult a healthcare professional
                Your answers must be in Turkish and maintain a caring, professional tone.
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

# --- Endpoints ---
@app.post("/chat", response_model=ChatOutput)
async def chat(
    chat_input: ChatInput,
    token: str = Depends(get_current_user_token)
):
    """
    Receives a question, a session_id, and a user's JWT.
    The user's baby_id is automatically fetched based on their token.
    If no baby context is available, AI operates in general mode providing general baby care advice.
    """
    session_id = chat_input.session_id or str(uuid.uuid4())

    # Fetch the baby_id for the authenticated user
    baby_id = get_baby_id_for_user(token)

    # Determine if we have baby context and can provide personalized responses
    has_baby_context = baby_id is not None

    report_text = ""
    if has_baby_context:
        report_text = get_baby_report(baby_id, token, chat_input.report_type)
        # If report fetch fails, fall back to general mode
        if report_text.startswith("Error:"):
            has_baby_context = False
            report_text = ""

    # Select appropriate prompt based on context availability
    if has_baby_context:
        prompt_template = get_context_prompt()
        context_info = "kişiselleştirilmiş"
    else:
        prompt_template = get_general_prompt()
        context_info = "genel bebek bakımı"

    # Fetch and save history using the user's token for security
    history_data = get_chat_history(session_id, token)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for message in history_data:
        if message['role'] == 'human':
            memory.chat_memory.add_user_message(message['message_content'])
        elif message['role'] == 'ai':
            memory.chat_memory.add_ai_message(message['message_content'])

    chain = prompt_template | llm | StrOutputParser()

    # Prepare response based on context availability
    if has_baby_context:
        response_text = chain.invoke({
            "report_text": report_text,
            "chat_history": memory.chat_memory.messages,
            "question": chat_input.question
        })
    else:
        response_text = chain.invoke({
            "chat_history": memory.chat_memory.messages,
            "question": chat_input.question
        })

    # Save the new exchange to history, also secured with the token
    add_to_chat_history(session_id, "human", chat_input.question, token)
    add_to_chat_history(session_id, "ai", response_text, token)

    return ChatOutput(response=response_text, session_id=session_id)


@app.get("/")
def read_root():
    return {"message": "Welcome to the BabyAI Chatbot API v4.0. Use /docs for API documentation."}

# Function to run the Uvicorn server
def start():
    """Starts the Uvicorn server."""
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start()
