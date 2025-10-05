from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple
from supabase import create_client, Client

# ===== Supabase Auth Client =====
def _get_supabase_auth_client(access_token: str) -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL or Key is not configured.")
    client = create_client(supabase_url, supabase_key)
    # Set authenticated user session (RLS)
    client.auth.set_session(access_token, "dummy_refresh_token")
    return client

# ===== Baby / User Helpers =====
def get_baby_id_for_user(access_token: str) -> Optional[str]:
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
        print(f"[get_baby_id_for_user] error: {e}")
        return None

# ===== Report Fetching (NEW SCHEMA ONLY) =====
_REPORT_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_REPORT_CACHE_TTL = 60  # seconds

def _parse_report_data(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}

def get_baby_report_raw(
    baby_id: str,
    access_token: str,
    report_type: str
) -> Optional[Dict[str, Any]]:
    """
    Gets the most recent report row (new schema). Cached briefly for performance.
    """
    key = (baby_id, report_type)
    now = time.time()
    cached = _REPORT_CACHE.get(key)
    if cached and (now - cached[0]) < _REPORT_CACHE_TTL:
        return cached[1]

    try:
        supabase = _get_supabase_auth_client(access_token)
        resp = (
            supabase.table("reports")
            .select("*")
            .eq("baby_id", baby_id)
            .eq("report_type", report_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data[0]
        parsed = _parse_report_data(row.get("data"))
        # Normalize expected structure
        categories = parsed.get("categories")
        aggregates = parsed.get("aggregates")
        meta = parsed.get("meta")
        if not isinstance(categories, dict):
            categories = {}
        if not isinstance(aggregates, dict):
            aggregates = {}
        if not isinstance(meta, dict):
            meta = {}
        row["data"] = {
            "categories": categories,
            "aggregates": aggregates,
            "meta": meta
        }
        _REPORT_CACHE[key] = (now, row)
        return row
    except Exception as e:
        print(f"[get_baby_report_raw] error: {e}")
        return None

def format_baby_report(
    report_row: Dict[str, Any],
    max_items_per_category: int = 5,
    include_aggregates: bool = True
) -> str:
    """
    Formats a report row (new schema).
    """
    if not report_row:
        return "Report: (empty)"
    rpt_type = report_row.get("report_type")
    created_at = report_row.get("created_at")
    data = report_row.get("data", {})
    categories: Dict[str, List[Dict[str, Any]]] = data.get("categories", {})
    aggregates: Dict[str, Any] = data.get("aggregates", {})
    meta: Dict[str, Any] = data.get("meta", {})

    lines: List[str] = []
    lines.append("=== BABY REPORT CONTEXT ===")
    lines.append(f"Type: {rpt_type}")
    if created_at:
        lines.append(f"DB Created: {created_at}")

    # Meta
    if meta:
        if meta.get("date"):
            lines.append(f"Date: {meta['date']}")
        elif meta.get("period_start") or meta.get("period_end"):
            lines.append(f"Period: {meta.get('period_start')} → {meta.get('period_end')}")
        if meta.get("log_count") is not None:
            lines.append(f"Logs Count: {meta['log_count']}")
        if meta.get("generated_at"):
            lines.append(f"Generated At: {meta['generated_at']}")
    lines.append("")

    # Aggregates
    if include_aggregates and aggregates:
        lines.append("[AGGREGATES]")
        for k, v in aggregates.items():
            if isinstance(v, dict):
                parts = ", ".join(f"{ik}={iv}" for ik, iv in v.items() if iv is not None)
                lines.append(f"- {k}: {parts}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    # Categories
    for cat, items in categories.items():
        lines.append(f"[{cat.upper()}]")
        if not isinstance(items, list) or not items:
            lines.append("- (no entries)")
            lines.append("")
            continue
        for item in items[:max_items_per_category]:
            it_type = item.get("type")
            it_data = item.get("data") or {}
            notes = it_data.get("notes")
            start = it_data.get("startTime")
            end = it_data.get("endTime")
            parts: List[str] = []
            if it_type:
                parts.append(it_type)
            # Durations
            if start and end:
                try:
                    s = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
                    e = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
                    diff = (e - s).total_seconds() / 60
                    if 0 < diff < 24 * 60:
                        parts.append(f"{int(round(diff))}m")
                except Exception:
                    pass
            if "value" in it_data:
                parts.append(f"value={it_data['value']}")
            if "temperature" in it_data:
                parts.append(f"temp={it_data['temperature']}°C")
            if "tooth_name" in it_data:
                parts.append(f"tooth={it_data['tooth_name']}")
            if notes:
                parts.append(notes[:120])
            if not parts:
                parts.append("entry")
            lines.append("- " + " | ".join(parts))
        if len(items) > max_items_per_category:
            lines.append(f"... (+{len(items) - max_items_per_category} more)")
        lines.append("")

    lines.append(f"(Trimmed to {max_items_per_category} items per category)")
    return "\n".join(lines)

def get_baby_report_formatted(
    baby_id: str,
    access_token: str,
    report_type: str,
    max_items_per_category: int = 5
) -> str:
    row = get_baby_report_raw(baby_id, access_token, report_type)
    if not row:
        return f"No report found: {report_type}"
    return format_baby_report(row, max_items_per_category=max_items_per_category)

def get_combined_reports_for_prompt(
    baby_id: str,
    access_token: str,
    include_weekly: bool = True,
    include_daily: bool = True,
    order: str = "weekly_first"
) -> str:
    blocks: List[Tuple[str, str]] = []
    if include_weekly:
        blocks.append(("weekly_summary",
                       get_baby_report_formatted(baby_id, access_token, "weekly_summary", 6)))
    if include_daily:
        blocks.append(("end_of_day_summary",
                       get_baby_report_formatted(baby_id, access_token, "end_of_day_summary", 6)))

    if order == "daily_first":
        blocks.sort(key=lambda x: 0 if x[0] == "end_of_day_summary" else 1)
    else:
        blocks.sort(key=lambda x: 0 if x[0] == "weekly_summary" else 1)

    # Join only existing (non-empty) blocks
    return "\n\n".join(b[1] for b in blocks if b[1] and not b[1].startswith("No report found"))

# ===== Chat History (değişmedi) =====
def get_chat_history(session_id: str, access_token: str):
    try:
        supabase = _get_supabase_auth_client(access_token)
        resp = (
            supabase.table("chat_history")
            .select("role, message_content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data if resp.data else []
    except Exception as e:
        print(f"[get_chat_history] error: {e}")
        return []

def add_to_chat_history(session_id: str, role: str, message: str, access_token: str):
    try:
        supabase = _get_supabase_auth_client(access_token)
        user = supabase.auth.get_user()
        if not user or not user.user:
            raise Exception("Invalid token: cannot fetch user.")
        supabase.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "message_content": message,
            "user_id": user.user.id,
        }).execute()
    except Exception as e:
        print(f"[add_to_chat_history] error: {e}")
