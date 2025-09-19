import uvicorn
import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from supabase_tools import get_baby_report, get_chat_history, add_to_chat_history

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Baby Care Chatbot API",
    description="A simple API for a baby care chatbot that remembers conversation history.",
    version="3.0.0",
)

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompt template with a placeholder for memory
prompt = ChatPromptTemplate.from_messages(
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

# Pydantic model for the request body
class ChatInput(BaseModel):
    question: str
    baby_id: str
    report_type: Optional[str] = "end_of_day_summary"
    session_id: Optional[str] = None

# Pydantic model for the response body
class ChatOutput(BaseModel):
    response: str
    session_id: str = Field(description="The unique identifier for the conversation session.")


# Define the /chat endpoint
@app.post("/chat", response_model=ChatOutput)
async def chat(chat_input: ChatInput):
    """
    Receives a baby_id, a question, and an optional session_id.
    1. Fetches the baby's report from Supabase.
    2. Fetches the conversation history for the session.
    3. Passes the report, history, and question to the LLM.
    4. Saves the new exchange to the history.
    5. Returns the answer and session_id.
    """
    # 1. Ensure a session ID exists
    session_id = chat_input.session_id or str(uuid.uuid4())

    # 2. Fetch the report from Supabase
    report_text = get_baby_report(chat_input.baby_id, chat_input.report_type)
    if report_text.startswith("Error:"):
        return ChatOutput(
            response="Üzgünüm, bebeğinizin raporunu alırken bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
            session_id=session_id
        )

    # 3. Fetch chat history and prepare memory
    history_data = get_chat_history(session_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for message in history_data:
        if message['role'] == 'human':
            memory.chat_memory.add_user_message(message['message_content'])
        elif message['role'] == 'ai':
            memory.chat_memory.add_ai_message(message['message_content'])

    # 4. Create and invoke the Langchain chain
    chain = prompt | llm | StrOutputParser()
    
    response_text = chain.invoke({
        "report_text": report_text,
        "chat_history": memory.chat_memory.messages,
        "question": chat_input.question
    })

    # 5. Save the new question and AI response to the history
    add_to_chat_history(session_id, "human", chat_input.question)
    add_to_chat_history(session_id, "ai", response_text)

    # 6. Return the answer and session_id
    return ChatOutput(response=response_text, session_id=session_id)


@app.get("/")
def read_root():
    return {"message": "Welcome to the BabyAI Chatbot API v3.0. Use the /chat endpoint to interact."}

# Function to run the Uvicorn server
def start():
    """Starts the Uvicorn server."""
    # Use port 8080 as the default, which is common for cloud deployments.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start()
