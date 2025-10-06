from __future__ import annotations
import os
from supabase import create_client, Client
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Client Initialization ---
def _get_supabase_auth_client(access_token: str) -> Client:
    """
    Creates a Supabase client authenticated with the user's JWT (for RLS).
    Uses service key to allow server-side access while RLS still applies with set_session.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")  # service role or anon with sufficient perms for RLS
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL or Key is not configured.")
    client = create_client(supabase_url, supabase_key)
    client.auth.set_session(access_token, "dummy_refresh_token")
    return client

# --- User/Baby Management ---
def get_baby_id_for_user(access_token: str) -> Optional[str]:
    """
    Returns the most recent (by created_at) baby id that belongs to the authenticated user.
    Expects babies table with column user_id referencing auth.uid().
    """
    try:
        supabase = _get_supabase_auth_client(access_token)
        user = supabase.auth.get_user()
        if not user or not user.user:
            return None
        user_id = user.user.id

        resp = (
            supabase.table("babies")
            .select("id")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data and resp.data[0].get("id"):
            return resp.data[0]["id"]
        return None
    except Exception as e:
        print(f"Error fetching baby_id for user: {e}")
        return None

# --- Chat History (requires user's access token) ---
def get_chat_history(session_id: str, access_token: str) -> List[Dict[str, Any]]:
    """Fetches chat history for a session, authenticated with the user's token (RLS enforced)."""
    try:
        supabase = _get_supabase_auth_client(access_token)
        response = (
            supabase.table("chat_history")
            .select("role, message_content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

def add_to_chat_history(session_id: str, role: str, message: str, access_token: str):
    """
    Adds a message to the chat history with the authenticated user's context.
    RLS on chat_history should enforce: user_id = auth.uid() (with check).
    """
    try:
        supabase = _get_supabase_auth_client(access_token)
        user = supabase.auth.get_user()
        if not user or not user.user:
            raise Exception("Invalid token, could not get user.")

        supabase.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "message_content": message,
            "user_id": user.user.id,
        }).execute()
    except Exception as e:
        print(f"Error saving chat history: {e}")

# --- Baby Logs (real-time context for chatbot) ---
def get_baby_logs_for_prompt(baby_id: str, access_token: str, limit: int = 100) -> str:
    """
    Fetches the most recent logs for a baby and formats them for LLM context.
    Returns real-time baby data without needing pre-generated reports.
    Also includes baby's age information for developmental context.
    """
    try:
        supabase = _get_supabase_auth_client(access_token)
        
        # First, get baby information (name, date of birth, gender)
        baby_info_response = (
            supabase.table("babies")
            .select("name, date_of_birth, gender")
            .eq("id", baby_id)
            .single()
            .execute()
        )
        
        baby_name = baby_info_response.data.get("name", "Bebek") if baby_info_response.data else "Bebek"
        date_of_birth = baby_info_response.data.get("date_of_birth") if baby_info_response.data else None
        gender = baby_info_response.data.get("gender", "bilinmiyor") if baby_info_response.data else "bilinmiyor"
        
        # Calculate age
        age_info = ""
        if date_of_birth:
            try:
                dob = datetime.fromisoformat(str(date_of_birth).replace("Z", "+00:00"))
                age_days = (datetime.now(dob.tzinfo) - dob).days
                age_months = age_days // 30
                age_weeks = age_days // 7
                
                if age_months >= 12:
                    age_years = age_months // 12
                    remaining_months = age_months % 12
                    age_info = f"{age_years} yaş {remaining_months} ay"
                elif age_months > 0:
                    age_info = f"{age_months} aylık"
                else:
                    age_info = f"{age_weeks} haftalık"
            except:
                age_info = "yaş bilgisi hesaplanamadı"
        
        # Fetch recent logs (ordered by creation time)
        response = (
            supabase.table("logs")
            .select("*")
            .eq("baby_id", baby_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        if not response.data:
            return f"=== BEBEK BİLGİLERİ ===\nİsim: {baby_name}\nYaş: {age_info}\nCinsiyet: {gender}\n\nHenüz hiç kayıt eklenmemiş."
        
        # Group logs by category
        logs_by_category = {}
        for log in reversed(response.data):  # Reverse to get chronological order (oldest → newest)
            category = log.get("category", "Diğer")
            if category not in logs_by_category:
                logs_by_category[category] = []
            logs_by_category[category].append(log)
        
        # Build formatted text for LLM
        prompt_lines = [
            "=== BEBEK BİLGİLERİ ===",
            f"İsim: {baby_name}",
            f"Yaş: {age_info}",
            f"Cinsiyet: {gender}",
            "",
            "=== BEBEK KAYITLARI ==="
        ]
        for category, logs in logs_by_category.items():
            prompt_lines.append(f"\n## {category.upper()} ##")
            for log in logs:
                log_type = log.get("type", "")
                log_data = log.get("data", {})
                created_at = log.get("created_at", "")
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    time_str = dt.strftime("%d.%m.%Y %H:%M")
                except:
                    time_str = created_at
                
                details = [f"Zaman: {time_str}"]
                if log_type:
                    details.append(f"Tür: {log_type}")
                
                # Add data fields
                if isinstance(log_data, dict):
                    for key, value in log_data.items():
                        if key in ["startTime", "endTime"]:
                            try:
                                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                                value = dt.strftime("%H:%M")
                            except:
                                pass
                        if value:
                            details.append(f"{key}: {value}")
                
                prompt_lines.append(f"- {' | '.join(details)}")
        
        return "\n".join(prompt_lines)
        
    except Exception as e:
        print(f"Error fetching baby logs: {e}")
        return f"Log verisi alınırken hata: {str(e)}"