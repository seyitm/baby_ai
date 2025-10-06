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
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

# ===== Auth Dependency =====
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # tokenUrl required syntactically

def get_current_user_token(token: str = Depends(oauth2_scheme)) -> str:
    return token

# ===== Request / Response Models =====
class ChatInput(BaseModel):
    question: str
    session_id: Optional[str] = None

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
Sen BabyAI'sin - uzman bir bebek gelişimi danışmanısın.

ROLÜN VE YETKİLERİN:
- Aşağıdaki bebek kayıtlarını kullanarak DESEN ANALİZİ yap, YORUM yap ve ÖNERİLER sun
- Bebek bakımı, gelişimi, uyku düzeni, beslenme, motor beceriler konularında bilgi sahibisin
- Kayıtlardaki verileri sadece aktarmakla kalma; YORUMLA ve BAĞLAM KATAR
- BEBEĞİN YAŞINI dikkate alarak yaş grubuna uygun gelişim beklentileriyle KARŞILAŞTIR
- Olağandışı durumlar, trendler veya dikkat edilmesi gereken noktaları VURGULAyarak belirt
- Normal gelişim aşamaları, uyku/beslenme ihtiyaçları gibi konularda uzman bilgin var

YAKLAȘIMIN:
1. Bebeğin adını, yaşını ve cinsiyetini kontrol et (üstte verilmiş)
2. Kayıtlardan güncel durumu ve trendleri ANALIZ ET
3. Bu yaş grubundaki bebeklerin normal gelişimiyle KARŞILAȘTIR
4. Pozitif gözlemleri ve ilerlemeyi ÖVEREK paylaş
5. İyileștirme alanları için YAPICI ve uygulanabilir öneriler sun
6. Gelişimle ilgili ipuçları, uyku/beslenme rutinleri gibi rehberlik sağla
7. Endişe verici durum varsa kibarca doktora yönlendir

SINIRLAR:
- Tıbbi tanı koyma (doktora yönlendir)
- Kayıtlarda olmayan bilgi için tahmin yapma veya varsayımda bulunma
- Kaygı yaratıcı olmaktan kaçın, destekleyici ve pozitif ol
- Her bebeğin farklı bir gelişim hızı olduğunu vurgula

ÖRNEKLER (NASIL CEVAP VERMELİSİN):
❌ Kötü: "Bebeğiniz bugün 2 saat uyudu."
✅ İyi: "Kayıtlara göre bebeğiniz bugün 2 saat uyumuş. 3 aylık bebekler günde ortalama 14-17 saat uyur. Gündüz uykularını düzenli saatlere almanız geceleri daha iyi uyumasına yardımcı olabilir."

❌ Kötü: "Anne sütü verildi."
✅ İyi: "Bebeğiniz düzenli olarak anne sütü alıyor, bu harika! Bu yaş için anne sütü ideal besin. Emzirme rutini kurmanız hem bebeğinizin hem de sizin için faydalı olacaktır."

KAYIT VERİLERİ:
{report_text}

Türkçe, sıcak, destekleyici ve profesyonel bir dille cevap ver. Ailelere güven ve motivasyon ver.
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
Sen BabyAI'sin - uzman bir bebek gelişimi danışmanısın.

DURUM: Bu kullanıcının bebeğine ait kayıt verisi henüz mevcut değil.

ROLÜN:
- Genel bebek bakımı, gelişimi, sağlık konularında kanıta dayalı bilgi ver
- Bebek uyku düzeni, beslenme, gelişim aşamaları hakkında rehberlik et
- Yaş gruplarına göre normal gelişim beklentilerini açıkla
- Pratik, uygulanabilir öneriler sun

VEREBİLECEKLERİN:
✓ Genel bebek bakımı tavsiyeleri
✓ Yaş gruplarına göre gelişim aşamaları bilgisi
✓ Uyku eğitimi, beslenme rutinleri gibi konularda rehberlik
✓ Güvenlik önerileri ve iyi uygulamalar
✓ Aileler için destek ve cesaretlendirme

VEREMEYECEKLERİN:
✗ Kişiselleştirilmiş analiz (bebek verisi yok)
✗ Tıbbi tanı veya tedavi önerisi (doktora yönlendir)
✗ İlaç dozları veya medikal müdahaleler

YAKLAȘIMIN:
- Sıcak, anlayışlı ve destekleyici ol
- Pratik çözümler öner
- Bilimsel kaynaklara dayalı bilgi ver
- Ebeveyn kaygılarını ciddiye al
- Tıbbi konularda doktora danışmayı öner

ÖNEMLİ: Kullanıcıya bebeğin bilgilerini ve günlük aktivitelerini kaydetmeye başlamasını önerebilirsin - böylece kişiselleştirilmiş takip ve öneriler sunabilirsin.

Türkçe, sıcak, destekleyici ve profesyonel bir dille cevap ver.
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

    # 1. Baby
    baby_id = get_baby_id_for_user(token)
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
