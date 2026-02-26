"""Streamlit frontend for the PDF RAG pipeline."""

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
REQUEST_TIMEOUT_QUERY = 120  # seconds — LLM call
POLL_INTERVAL = 1.0          # seconds — progress bar polling interval

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Inter font + sage green palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background-color: #f7faf8;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #eef3ee;
    border-right: 1px solid #ccdccc;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Primary button (Index Document) ── */
div.stButton > button[kind="primary"] {
    background-color: #3d6b4f !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.73rem !important;
    padding: 0.28rem 1rem !important;
    height: auto !important;
    min-height: unset !important;
    white-space: nowrap !important;
    transition: background-color 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #2f5540 !important;
}
/* Centre the primary button in the sidebar */
section[data-testid="stSidebar"] div.stButton {
    display: flex !important;
    justify-content: center !important;
}

/* ── Sidebar non-primary buttons — red, transparent (delete + clear chat) ── */
section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #dc2626 !important;
    padding: 0 0.25rem !important;
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
    height: auto !important;
    min-height: unset !important;
}
section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
    background-color: #fef2f2 !important;
    background: #fef2f2 !important;
    border: none !important;
    color: #b91c1c !important;
}

/* ── File uploader — compact, smaller text ── */
[data-testid="stFileUploaderDropzone"] {
    padding: 0.5rem 0.6rem !important;
    border-radius: 8px !important;
    background: transparent !important;
}
/* Hide default drag-and-drop instructions (replaced by custom text above) */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
/* Browse files button — small, with paperclip */
[data-testid="stFileUploaderDropzone"] button {
    background: white !important;
    border: 1px solid #ccdccc !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
    padding: 0.2rem 0.75rem !important;
    color: #3d6b4f !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzone"] button::before {
    content: "📎";
    margin-right: 0.3rem;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: #3d6b4f !important;
    background: #f0f8f2 !important;
}
/* Uploaded file row — vertically align icon, name, X button */
[data-testid="stFileUploaderFile"] {
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    padding: 0.25rem 0 !important;
}
[data-testid="stFileUploaderFile"] svg {
    width: 1rem !important;
    height: 1rem !important;
    flex-shrink: 0 !important;
}
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFileName"] {
    font-size: 0.73rem !important;
    line-height: 1.2 !important;
}
/* X dismiss button inside file row */
[data-testid="stFileUploaderDeleteBtn"] button,
[data-testid="stFileUploaderFile"] > button {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0.1rem !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    color: #6b8f6b !important;
}
[data-testid="stFileUploaderDeleteBtn"] button svg,
[data-testid="stFileUploaderFile"] > button svg {
    width: 0.85rem !important;
    height: 0.85rem !important;
}

/* ── Sidebar selectbox — smaller (selected value + dropdown options) ── */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    border-color: #ccdccc !important;
    border-radius: 8px !important;
    min-height: 28px !important;
    padding-top: 0.1rem !important;
    padding-bottom: 0.1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    font-size: 0.75rem !important;
}
/* Dropdown option list */
ul[data-baseweb="menu"] li,
ul[data-baseweb="menu"] li span {
    font-size: 0.75rem !important;
}

/* ── Progress bar fill ── */
[data-testid="stProgressBar"] > div > div {
    background-color: #3d6b4f !important;
}

/* ── Chat input — remove white oval container background ── */
.stChatInputContainer > div,
[data-testid="stBottomBlockContainer"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea {
    border: 1px solid #ccdccc !important;
    border-radius: 12px !important;
    background-color: #ffffff !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3d6b4f !important;
    box-shadow: 0 0 0 1px #3d6b4f !important;
}

/* ── Headings ── */
h1, h2, h3 {
    color: #1c2e1c !important;
}

/* ── Dividers ── */
hr {
    border-color: #ccdccc !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def format_citation(source: dict, index: int) -> str:
    pages = ", ".join(f"p.{p}" for p in source["page_numbers"]) if source.get("page_numbers") else "?"
    label = source.get("heading") or (
        " > ".join(source["section_hierarchy"])
        if source.get("section_hierarchy")
        else source.get("filename", "?")
    )
    return f"**[{index}]** {source.get('filename', '?')} — {label} ({pages}) · score: {source['score']:.3f}"


def session_doc_ids() -> list[str]:
    return [d["document_id"] for d in st.session_state.session_docs]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 PDF RAG Assistant")
    st.divider()

    # ── Upload section ──────────────────────────────────────────────────────
    st.markdown("**Upload Document**")
    st.markdown(
        '<p style="font-size:0.72rem; color:#6b8f6b; margin:-0.3rem 0 0.4rem 0;">'
        "Drag and drop PDF file here (200 MB max.)</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        index_clicked = st.button("Index Document", type="primary")

        if index_clicked:
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=30,
                )

                if resp.ok:
                    job_data = resp.json()
                    job_id = job_data["job_id"]
                    filename = job_data["filename"]

                    progress_bar = st.progress(0.0, text=f"📄 {filename} — starting…")
                    status: dict = {}

                    while True:
                        try:
                            poll = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5)
                            if poll.ok:
                                status = poll.json()
                                progress_bar.progress(
                                    min(float(status.get("progress", 0.0)), 1.0),
                                    text=status.get("stage", "Processing…"),
                                )
                                if status.get("status") in ("done", "error"):
                                    break
                        except requests.exceptions.ConnectionError:
                            break
                        time.sleep(POLL_INTERVAL)

                    progress_bar.empty()

                    if status.get("status") == "done":
                        doc = {
                            "document_id": status["document_id"],
                            "filename": status["filename"],
                            "total_chunks": status["total_chunks"] or 0,
                            "text_chunks": status["text_chunks"] or 0,
                            "table_chunks": status["table_chunks"] or 0,
                            "image_chunks": status["image_chunks"] or 0,
                        }
                        existing_ids = {d["document_id"] for d in st.session_state.session_docs}
                        if doc["document_id"] not in existing_ids:
                            st.session_state.session_docs.append(doc)
                            if status.get("already_indexed"):
                                st.info(f"**{filename}** was already indexed — added to your session.")
                            else:
                                st.success(f"Indexed **{filename}** successfully.")
                        else:
                            st.info(f"**{filename}** is already in your session.")
                        st.rerun()
                    else:
                        st.error(f"Indexing failed: {status.get('error', 'Unknown error')}")

                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is the backend running?\n```\nuvicorn src.api.main:app --reload\n```")
            except requests.exceptions.Timeout:
                st.error("Request timed out while starting the job.")

    st.divider()

    # ── Indexed documents (session-only) ────────────────────────────────────
    st.markdown("**My Documents**")

    if not st.session_state.session_docs:
        st.caption("No documents yet. Upload a PDF above.")
    else:
        for doc in list(st.session_state.session_docs):
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(
                    f'<p style="font-size:0.75rem; color:#2d4a2d; margin:0.15rem 0;">📄 {doc["filename"]}</p>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑", key=f"del_{doc['document_id']}", help="Delete from system"):
                    try:
                        r = requests.delete(
                            f"{API_URL}/documents/{doc['document_id']}", timeout=15
                        )
                        if r.ok:
                            st.session_state.session_docs = [
                                d for d in st.session_state.session_docs
                                if d["document_id"] != doc["document_id"]
                            ]
                            if st.session_state.selected_doc_id == doc["document_id"]:
                                st.session_state.selected_doc_id = None
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {r.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach the API.")

    st.divider()

    # ── Query scope ─────────────────────────────────────────────────────────
    if st.session_state.session_docs:
        st.markdown("**Query scope**")
        doc_options = {"All my documents": None} | {
            d["filename"]: d["document_id"] for d in st.session_state.session_docs
        }
        selected_label = st.selectbox(
            "Filter to:",
            options=list(doc_options.keys()),
            label_visibility="collapsed",
        )
        st.session_state.selected_doc_id = doc_options[selected_label]
        st.divider()

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — welcome screen or chat
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 15vh;
            text-align: center;
        ">
            <h1 style="
                font-size: 2.6rem;
                font-weight: 700;
                color: #1c2e1c;
                margin-bottom: 0.4rem;
                letter-spacing: -0.5px;
            ">PDF RAG Assistant</h1>
            <p style="
                font-size: 1.05rem;
                color: #5a7a5a;
                max-width: 460px;
                line-height: 1.6;
                margin: 0;
            ">Upload a PDF in the sidebar, then ask questions about its content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(format_citation(src, i))

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if question := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.session_docs:
        st.warning("Please upload at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        selected = st.session_state.selected_doc_id
        document_ids = [selected] if selected else session_doc_ids()

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": question,
                            "n_results": 5,
                            "document_ids": document_ids,
                        },
                        timeout=REQUEST_TIMEOUT_QUERY,
                    )

                    if resp.ok:
                        data = resp.json()
                        st.markdown(data["answer"])

                        if data["sources"]:
                            with st.expander(f"Sources ({len(data['sources'])})"):
                                for i, src in enumerate(data["sources"], 1):
                                    st.markdown(format_citation(src, i))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"],
                        })
                    else:
                        error_msg = f"Error {resp.status_code}: {resp.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except requests.exceptions.Timeout:
                    msg = "The request timed out. Try a simpler question or check the backend."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                except requests.exceptions.ConnectionError:
                    msg = "Cannot reach the API. Is the backend running?"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
