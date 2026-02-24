"""Supabase-backed usage tracking for Parserix upload limits."""

import streamlit as st
from supabase import create_client, Client

MAX_UPLOADS = 10


@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and cache the Supabase client from Streamlit secrets."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            "Supabase credentials not configured. "
            "Add SUPABASE_URL and SUPABASE_KEY to your Streamlit secrets."
        ) from e
    return create_client(url, key)


def get_upload_count(email: str) -> int:
    """Return the total upload count for a given email. Returns 0 if not found."""
    sb = get_supabase_client()
    response = (
        sb.table("usage")
        .select("upload_count")
        .eq("user_email", email.lower().strip())
        .execute()
    )
    if response.data:
        return response.data[0]["upload_count"]
    return 0


def increment_upload_count(email: str, count: int = 1) -> int:
    """Increment the upload count for a user. Creates the row if it doesn't exist.

    Returns the new total upload count.
    """
    email = email.lower().strip()
    sb = get_supabase_client()

    current = get_upload_count(email)

    if current == 0:
        # Check if row exists with 0 count vs doesn't exist
        response = (
            sb.table("usage")
            .select("id")
            .eq("user_email", email)
            .execute()
        )
        if not response.data:
            # Insert new row
            sb.table("usage").insert({
                "user_email": email,
                "upload_count": count,
            }).execute()
            return count

    # Update existing row
    new_count = current + count
    sb.table("usage").update({
        "upload_count": new_count,
        "updated_at": "now()",
    }).eq("user_email", email).execute()
    return new_count


def get_remaining_quota(email: str) -> int:
    """Return how many uploads the user has left."""
    return max(0, MAX_UPLOADS - get_upload_count(email))
