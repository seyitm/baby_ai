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

# --- Baby Report (requires user context) ---
def get_baby_report(baby_id: str, access_token: str, report_type: Optional[str] = "end_of_day_summary") -> str:
    """
    Fetches the latest report row for a given baby from 'reports'.
    RLS on 'reports' must ensure auth.uid() is allowed for the given baby_id.
    """
    try:
        supabase: Client = _get_supabase_auth_client(access_token)

        query = supabase.table("reports").select("*").eq("baby_id", baby_id)
        if report_type:
            query = query.eq("report_type", report_type)

        response = query.order("created_at", desc=True).limit(1).execute()
        if not response.data:
            return f"Error: No report found for baby {baby_id} with type {report_type}."

        report_data = response.data[0]

        # Format the report data into a readable string for the LLM
        report_details = ["Here is the baby's report:"]
        for key, value in report_data.items():
            if value is None or key in ["baby_id", "id"]:
                continue

            if key == "data" and isinstance(value, dict):
                for category, items in value.items():
                    if isinstance(items, list) and items:
                        report_details.append(f"\n## {category} Summary ##")
                        for item in items:
                            details = []
                            item_data = item.get("data", {})

                            if item.get("type"):
                                details.append(f"Type: {item['type']}")

                            start_str = item_data.get("startTime")
                            end_str = item_data.get("endTime")
                            if start_str and end_str:
                                try:
                                    start_time = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
                                    end_time = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
                                    duration_minutes = round((end_time - start_time).total_seconds() / 60)
                                    details.append(f"Start: {start_time.strftime('%H:%M')}")
                                    details.append(f"End: {end_time.strftime('%H:%M')}")
                                    details.append(f"Duration: {duration_minutes} minutes")
                                except (ValueError, TypeError):
                                    pass

                            if item_data.get("notes"):
                                details.append(f"Notes: {item_data['notes']}")

                            if details:
                                report_details.append(f"- {' | '.join(details)}")
            else:
                formatted_key = key.replace("_", " ").capitalize()
                report_details.append(f"- {formatted_key}: {value}")

        return "\n".join(report_details)

    except Exception as e:
        return f"Error: An error occurred while fetching the report from Supabase: {e}"

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