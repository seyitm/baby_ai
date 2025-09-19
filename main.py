import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

from supabase_tools import get_baby_report

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Baby Care Chatbot API",
    description="A simple API for a baby care chatbot that answers questions based on a pre-fetched report.",
    version="2.0.0",
)

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompt template
# It takes the fetched report and the user's question as input
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are BabyAI, a helpful and caring baby care assistant.
            You will be given a detailed report containing all available information about a specific baby.
            Your task is to answer the user's question based ONLY on the information provided in that report.
            Do not make up information. If the answer is not in the report, politely say that you do not have that specific information in the report.
            Your answers must be in Turkish.
            """,
        ),
        ("human", "Here is the baby's report:\n\n{report_text}\n\n---\n\nMy Question is: {question}"),
    ]
)

# Define the output parser
output_parser = StrOutputParser()

# Create the simple Langchain chain
chain = prompt | llm | output_parser

# Pydantic model for the request body
class ChatInput(BaseModel):
    question: str
    baby_id: str
    report_type: Optional[str] = "end_of_day_summary"

# Define the /chat endpoint
@app.post("/chat")
async def chat(chat_input: ChatInput):
    """
    Receives a baby_id and a question.
    1. Fetches the baby's report from Supabase.
    2. Passes the report and question to the LLM.
    3. Returns the answer.
    """
    # 1. Fetch the report from Supabase
    report_text = get_baby_report(chat_input.baby_id, chat_input.report_type)

    # Handle cases where the report could not be fetched
    if report_text.startswith("Error:"):
        # You can customize this error message
        print(f"DEBUG: An error occurred while fetching the report: {report_text}")
        return {"response": "Üzgünüm, bebeğinizin raporunu alırken bir sorun oluştu. Lütfen daha sonra tekrar deneyin."}

    # 2. Pass the report and question to the chain
    response = chain.invoke(
        {"report_text": report_text, "question": chat_input.question}
    )
    
    # 3. Return the answer
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Welcome to the BabyAI Chatbot API v2.0. Use the /chat endpoint to interact."}

# Function to run the Uvicorn server
def start():
    """Starts the Uvicorn server."""
    # Use port 8080 as the default, which is common for cloud deployments.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start()
