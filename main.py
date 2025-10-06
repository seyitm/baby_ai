import os
import uuid
import uvicorn
from typing import Optional

from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from supabase_tools import (
    get_baby_id_for_user,
    get_baby_logs_for_prompt,
    get_chat_history,
    add_to_chat_history
)

# ===== Environment Load =====
load_dotenv()

# ===== FastAPI App =====
app = FastAPI(
    title="Baby Care Chatbot API",
    description="A secure API for a baby care chatbot with real-time log context.",
    version="5.1.0",
)

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8081,https://babyai-production.up.railway.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "*"],
)

# ===== LLM Init =====
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "350"))
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.6"))
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    max_output_tokens=MODEL_MAX_TOKENS,
    temperature=MODEL_TEMPERATURE,
)

# ===== Auth Dependency =====
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # tokenUrl required syntactically

def get_current_user_token(token: str = Depends(oauth2_scheme)) -> str:
    return token

# ===== Request / Response Models =====
class ChatInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    baby_id: Optional[str] = None

class ChatOutput(BaseModel):
    response: str
    session_id: str = Field(description="Conversation session id")

# ===== Prompt Builders =====
def get_context_prompt() -> ChatPromptTemplate:
    """
    Kişiselleştirilmiş mod – bebek kayıtları (logs) mevcut.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Sen BabyAI'sin. Kısa ve öz yanıt ver:
- En fazla 4 madde ya da 3–5 kısa cümle yaz.
- Yalnızca aşağıdaki BEBEK KAYITLARI ve sohbet geçmişine dayan.
- Kayıtlarda yoksa "Bu bilgi elimde yok." de.
- Tıbbi tanı verme; risk görürsen kibarca doktora yönlendir.

BEBEK KAYITLARI:
{report_text}
                """.strip()
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

def get_general_prompt() -> ChatPromptTemplate:
    """
    Genel mod – bebek kayıtları yok (henüz bebek eklenmemiş veya log yok).
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Sen BabyAI'sin. Kısa ve pratik yanıt ver:
- En fazla 4 madde ya da 3–5 kısa cümle.
- Kanıta dayalı, yaşa uygun genel öneriler ver; kişisel veri yok.
- Tıbbi tanı verme; risk görürsen kibarca doktora yönlendir.
                """.strip()
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

# ===== Helpers =====
MAX_HISTORY_MESSAGES = 20  # son N mesajı tut (user+ai toplam)
def _build_memory(session_id: str, token: str) -> ConversationBufferMemory:
    history = get_chat_history(session_id, token)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    for row in trimmed:
        role = row.get("role")
        content = row.get("message_content", "")
        if not content:
            continue
        if role == "human":
            memory.chat_memory.add_user_message(content)
        elif role == "ai":
            memory.chat_memory.add_ai_message(content)
    return memory

# ===== Chat Endpoint =====
@app.post("/chat", response_model=ChatOutput)
async def chat(
    chat_input: ChatInput,
    token: str = Depends(get_current_user_token)
):
    """
    Ana sohbet endpoint'i.
    Bebeğin gerçek zamanlı log kayıtlarını (mevcutsa) ekler, yoksa genel moda düşer.
    """
    session_id = chat_input.session_id or str(uuid.uuid4())

    # 1. Baby selection: trust RLS. Use provided baby_id if any; otherwise fallback to latest user's baby
    baby_id = chat_input.baby_id or get_baby_id_for_user(token)
    has_baby_context = baby_id is not None

    # 2. Loglar (gerçek zamanlı bebek kayıtları)
    report_text = ""
    if has_baby_context:
        try:
            report_text = get_baby_logs_for_prompt(
                baby_id=baby_id,
                access_token=token,
                limit=100  # Son 100 log kaydı
            )
            if not report_text.strip():
                has_baby_context = False
        except Exception as e:
            print(f"[chat] log fetch error: {e}")
            has_baby_context = False
            report_text = ""

    # 3. Geçmiş / hafıza
    memory = _build_memory(session_id, token)

    # 4. Prompt seçimi
    if has_baby_context:
        prompt_template = get_context_prompt()
    else:
        prompt_template = get_general_prompt()

    chain = prompt_template | llm | StrOutputParser()

    # 5. Model çağrısı
    if has_baby_context:
        variables = {
            "report_text": report_text,
            "chat_history": memory.chat_memory.messages,
            "question": chat_input.question
        }
    else:
        variables = {
            "chat_history": memory.chat_memory.messages,
            "question": chat_input.question
        }

    try:
        response_text = chain.invoke(variables)
    except Exception as e:
        print(f"[chat] LLM invocation error: {e}")
        response_text = (
            "Şu anda yanıt oluştururken bir sorun yaşadım. Lütfen tekrar dener misiniz?"
        )

    # 6. Kayıt (history)
    try:
        add_to_chat_history(session_id, "human", chat_input.question, token)
        add_to_chat_history(session_id, "ai", response_text, token)
    except Exception as e:
        # History yazılamazsa chat yine de dönsün
        print(f"[chat] history save error: {e}")

    return ChatOutput(response=response_text, session_id=session_id)

# ===== Root =====
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the BabyAI Chatbot API v5.1. Gerçek zamanlı log destekli /chat endpoint kullanın."
    }

# ===== Run (dev) =====
def start():
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    start()
