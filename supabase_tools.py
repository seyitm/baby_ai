from __future__ import annotations
import os
from supabase import create_client, Client
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Client Initialization ---
def _get_supabase_auth_client(access_token: str) -> Client:
    """Creates a Supabase client authenticated with the user's JWT."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL or Key is not configured.")
    
    client = create_client(supabase_url, supabase_key)
    client.auth.set_session(access_token, "dummy_refresh_token")
    return client

# --- User/Baby Management ---
def get_baby_id_for_user(access_token: str) -> Optional[str]:
    """Fetches the baby_id associated with the current user."""
    try:
        supabase = _get_supabase_auth_client(access_token)
        user = supabase.auth.get_user()
        if not user or not user.user:
            return None
        
        user_id = user.user.id
        response = supabase.table("users").select("baby_id").eq("id", user_id).limit(1).execute()

        if response.data and response.data[0].get("baby_id"):
            return response.data[0]["baby_id"]
        return None
    except Exception as e:
        print(f"Error fetching baby_id for user: {e}")
        return None

# --- Baby Report (Now requires user context) ---
def get_baby_report(baby_id: str, access_token: str, report_type: Optional[str] = "end_of_day_summary") -> str:
    """
    Fetches the latest report for a given baby.
    SECURITY: This function now requires a user token. You must set up RLS policies
    in Supabase to ensure the user is authorized to view this baby's report.
    """
    try:
        supabase: Client = _get_supabase_auth_client(access_token)

        # Fetch the single report row for the given baby_id
        # The RLS policy on the 'reports' table should verify access.
        query = supabase.table("reports").select("*").eq("baby_id", baby_id)

        if report_type:
            query = query.eq("report_type", report_type)

        response = query.order("created_at", desc=True).limit(1).execute()
        
        if not response.data:
            return f"Error: No report found for baby {baby_id} with type {report_type}."
            
        report_data = response.data[0]
        
        # Format the report data into a clean, human-readable string for the LLM
        report_details = ["Here is the baby's report:"]
        
        for key, value in report_data.items():
            if value is None or key in ["baby_id", "id"]:
                continue

            if key == 'data' and isinstance(value, dict):
                for category, items in value.items():
                    if isinstance(items, list) and items:
                        report_details.append(f"\n## {category} Summary ##")
                        for item in items:
                            details = []
                            item_data = item.get("data", {})
                            
                            # Type
                            if item.get("type"):
                                details.append(f"Type: {item['type']}")

                            # Duration and Times
                            start_str = item_data.get("startTime")
                            end_str = item_data.get("endTime")
                            
                            if start_str and end_str:
                                try:
                                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                                    duration = end_time - start_time
                                    duration_minutes = round(duration.total_seconds() / 60)
                                    
                                    details.append(f"Start: {start_time.strftime('%H:%M')}")
                                    details.append(f"End: {end_time.strftime('%H:%M')}")
                                    details.append(f"Duration: {duration_minutes} minutes")
                                except (ValueError, TypeError):
                                    pass # Skip if dates are not in the correct format

                            # Notes
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


# --- Chat History (Now requires user's access token) ---
def get_chat_history(session_id: str, access_token: str) -> List[Dict[str, Any]]:
    """Fetches chat history for a session, authenticated with the user's token."""
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
    """Adds a message to the chat history, authenticated with the user's token."""
    try:
        supabase = _get_supabase_auth_client(access_token)
        # RLS policy in Supabase will enforce that auth.uid() matches the user_id
        # We need to add the user_id column from the JWT on Supabase side,
        # or pass it here. For now, assuming RLS will handle it if `user_id` is set to `auth.uid()` on insert.
        # A trigger in Supabase is the best way to do this.
        # For now, we rely on the RLS `with check` clause.
        # Let's add an RLS function in Supabase to set the user_id on insert.
        
        # The user_id is NOT automatically inferred. We need to set it.
        # The RLS policy *enforces* it, but doesn't set it.
        # We need to decode the JWT to get the user_id.
        # Let's adjust.
        
        user = supabase.auth.get_user()
        if not user:
             raise Exception("Invalid token, could not get user.")

        supabase.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "message_content": message,
            "user_id": user.user.id
        }).execute()
    except Exception as e:
        print(f"Error saving chat history: {e}")
