import streamlit as st
import json

st.title("🔑 GCP Service Account Key Test")

try:
    # Access service account secrets
    sa_block = st.secrets["gcp_service_account"]

    # If using JSON style (service_account_json field)
    if "service_account_json" in sa_block:
        sa_dict = json.loads(sa_block["service_account_json"])
    else:
        # If using field-by-field style
        sa_dict = dict(sa_block)

    # Display basic fields to confirm key loads
    st.write("✅ Service Account loaded successfully")
    st.write("Project ID:", sa_dict.get("project_id", "Not found"))
    st.write("Client Email:", sa_dict.get("client_email", "Not found"))
    st.write("Private Key ID:", sa_dict.get("private_key_id", "Not found"))

except Exception as e:
    st.error(f"❌ Error loading service account: {e}")
