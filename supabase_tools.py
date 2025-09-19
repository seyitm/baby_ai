import os
from supabase import create_client, Client
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

def get_baby_report(baby_id: str, report_type: Optional[str] = "end_of_day_summary") -> str:
    """
    Fetches the latest report for a given baby from the 'reports' table.
    It can filter by report_type and gets the most recent one.
    """
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            return "Error: Supabase URL or Key is not configured."

        supabase: Client = create_client(supabase_url, supabase_key)

        # Fetch the single report row for the given baby_id
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

def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """Fetches chat history for a given session ID from Supabase."""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            return [] # Return empty list if Supabase is not configured

        supabase: Client = create_client(supabase_url, supabase_key)
        response = (
            supabase.table("chat_history")
            .select("role, message_content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data if response.data else []
    except Exception:
        # In case of an error, return an empty history to not break the chat
        return []

def add_to_chat_history(session_id: str, role: str, message: str):
    """Adds a new message to the chat history in Supabase."""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            return

        supabase: Client = create_client(supabase_url, supabase_key)
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "message_content": message
        }).execute()
    except Exception as e:
        # Log the error but don't crash the application
        print(f"Error saving chat history: {e}")
