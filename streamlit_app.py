import streamlit as st
import pandas as pd
import os
import time
from documents import get_documents, download_pdf_from_url, load_pdf, safe_pdf_filename
from chroma_store import add_document_to_store, initialize_store, COLLECTION_NAME
from retrieve_context import retrieve_context
from prompter import get_answer
import config

st.set_page_config(page_title="Nuclear Law Assistant", page_icon="⚛️", layout="wide")

# ─── Session State ────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    docs = get_documents()
    st.session_state.knowledge_base = pd.DataFrame(docs)
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "queued_question" not in st.session_state:
    st.session_state.queued_question = None

# ─── Theme Configuration ───────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg-base": "#080c14", "bg-surface": "#0d1420", "bg-elevated": "#121b2e",
        "bg-card": "#162035", "bg-hover": "#1c2940",
        "border": "rgba(255,255,255,0.08)", "border-accent": "rgba(0,212,255,0.28)",
        "grad-start": "#00d4ff", "grad-end": "#7b2fff",
        "text-primary": "#e8f0fe", "text-secondary": "#8a9cc0", "text-muted": "#5a6f90",
        "text-accent": "#00d4ff",
        "green": "#22c55e", "amber": "#f59e0b", "red": "#ef4444",
        "code-bg": "rgba(0,0,0,0.35)",
        "shadow": "rgba(0,0,0,0.45)",
    },
    "light": {
        "bg-base": "#f3f5fa", "bg-surface": "#ffffff", "bg-elevated": "#eef1f8",
        "bg-card": "#ffffff", "bg-hover": "#eef4fb",
        "border": "rgba(15,23,42,0.10)", "border-accent": "rgba(8,145,178,0.45)",
        "grad-start": "#0891b2", "grad-end": "#7c3aed",
        "text-primary": "#101827", "text-secondary": "#48566b", "text-muted": "#8a97ab",
        "text-accent": "#0891b2",
        "green": "#16a34a", "amber": "#b45309", "red": "#dc2626",
        "code-bg": "rgba(8,145,178,0.06)",
        "shadow": "rgba(15,23,42,0.10)",
    },
}

ATOM_SVG = """<svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
<circle cx="20" cy="20" r="18" stroke="url(#atomgrad)" stroke-width="2.4"/>
<circle cx="20" cy="20" r="4" fill="url(#atomgrad)"/>
<ellipse cx="20" cy="20" rx="14" ry="5" stroke="url(#atomgrad)" stroke-width="1.5"/>
<ellipse cx="20" cy="20" rx="14" ry="5" stroke="url(#atomgrad)" stroke-width="1.5" transform="rotate(60 20 20)"/>
<ellipse cx="20" cy="20" rx="14" ry="5" stroke="url(#atomgrad)" stroke-width="1.5" transform="rotate(120 20 20)"/>
<defs><linearGradient id="atomgrad" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
<stop stop-color="#00d4ff"/><stop offset="1" stop-color="#7b2fff"/></linearGradient></defs>
</svg>"""

SAMPLE_QUESTIONS = [
    "What is the NPT and what are its main provisions?",
    "Explain the difference between CSA and Additional Protocol.",
    "What are the key principles of nuclear liability?",
    "How does the IAEA verify compliance with non-proliferation?",
    "What is the role of the IAEA Statute in nuclear law?",
    "Explain the concept of 'strict liability' in nuclear law.",
]

WELCOME_CARDS = [
    ("📜", "Legal Frameworks", "NPT · IAEA Statute · CSA · Additional Protocol"),
    ("⚖️", "Nuclear Liability", "Strict liability, compensation & state responsibility"),
    ("🔍", "Compliance & Verification", "How the IAEA verifies non-proliferation commitments"),
    ("📤", "Custom Documents", "Upload your own PDFs or add a URL to extend the knowledge base"),
]


# ─── Theming (CSS) ──────────────────────────────────────────────────────────────
def inject_theme_css():
    t = THEMES[st.session_state.theme]

    themed = """
    :root {{
        --bg-base: {bg_base};
        --bg-surface: {bg_surface};
        --bg-elevated: {bg_elevated};
        --bg-card: {bg_card};
        --bg-hover: {bg_hover};
        --border: {border};
        --border-accent: {border_accent};
        --grad-start: {grad_start};
        --grad-end: {grad_end};
        --grad: linear-gradient(135deg, {grad_start}, {grad_end});
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --text-accent: {text_accent};
        --green: {green};
        --amber: {amber};
        --red: {red};
        --code-bg: {code_bg};
        --shadow-color: {shadow};
        --primary-color: var(--grad-start);
        --background-color: var(--bg-base);
        --secondary-background-color: var(--bg-card);
        --text-color: var(--text-primary);
        --font: 'Inter','Segoe UI',sans-serif;
    }}
    """.format(
        bg_base=t["bg-base"], bg_surface=t["bg-surface"], bg_elevated=t["bg-elevated"],
        bg_card=t["bg-card"], bg_hover=t["bg-hover"], border=t["border"],
        border_accent=t["border-accent"], grad_start=t["grad-start"], grad_end=t["grad-end"],
        text_primary=t["text-primary"], text_secondary=t["text-secondary"], text_muted=t["text-muted"],
        text_accent=t["text-accent"], green=t["green"], amber=t["amber"], red=t["red"],
        code_bg=t["code-bg"], shadow=t["shadow"],
    )

    static = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: var(--font); }
    * { scrollbar-width: thin; scrollbar-color: var(--border-accent) transparent; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.35); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-accent); }

    .stApp { background: var(--bg-base) !important; color: var(--text-primary) !important; }
    .main > div { background: var(--bg-base) !important; }
    [data-testid="stAppViewContainer"] { background: var(--bg-base) !important; }
    [data-testid="stHeader"] { background: transparent !important; }

    /* Center the chat column like ChatGPT / Claude */
    .block-container {
        max-width: 900px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 6rem !important;
        margin: 0 auto !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 320px !important;
        max-width: 320px !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] .block-container { max-width: 100% !important; padding: 0 0.9rem !important; }

    .sidebar-logo {
        display: flex; align-items: center; gap: 10px;
        padding: 2px 2px 14px 2px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }
    .sidebar-logo .logo-icon {
        font-size: 1.7rem; filter: drop-shadow(0 0 8px rgba(0,212,255,.30));
    }
    .sidebar-logo .logo-title {
        font-weight: 700; font-size: 1.05rem; line-height: 1.1;
        background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-logo .logo-subtitle {
        font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase;
    }

    .sidebar-section { margin-bottom: 1.15rem; }
    .section-title {
        display: flex; align-items: center; gap: 6px;
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.55rem;
    }

    /* Status card */
    .status-badge {
        display: flex; align-items: center; gap: 10px;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 9px 12px; margin-bottom: 1.1rem;
        box-shadow: 0 2px 10px var(--shadow-color);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; box-shadow: 0 0 6px var(--green); }
    .status-dot.empty { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
    .status-title { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); }
    .status-sub { font-size: 0.7rem; color: var(--text-muted); }

    /* Doc items */
    .doc-item {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 8px 11px; margin-bottom: 6px;
        transition: border-color .2s, background .2s;
    }
    .doc-item:hover { border-color: var(--border-accent); background: var(--bg-hover); }
    .doc-item .doc-name { font-size: 0.8rem; font-weight: 500; color: var(--text-primary); word-break: break-word; }
    .doc-item .doc-meta {
        display: inline-block; margin-top: 3px; font-size: 0.62rem; font-weight: 600;
        letter-spacing: .04em; text-transform: uppercase; color: var(--text-accent);
        background: rgba(0,212,255,0.10); border: 1px solid var(--border-accent);
        border-radius: 100px; padding: 1px 8px;
    }

    /* Sample question buttons */
    .stButton > button[kind="secondary"],
    div[data-testid="stVerticalBlock"] .sample-q-wrap .stButton > button {
        width: 100%; text-align: left; background: transparent;
        border: 1px solid var(--border); border-radius: 9px;
        color: var(--text-secondary); font-size: 0.78rem; font-weight: 400;
        padding: 7px 11px; white-space: normal; word-wrap: break-word;
        transition: all .2s; box-shadow: none;
    }

    /* New Chat button (sidebar, prominent) */
    .new-chat-btn button {
        width: 100%; background: var(--grad) !important; color: #fff !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: 0.55rem 0 !important;
        box-shadow: 0 3px 14px rgba(0,212,255,0.22);
        margin-bottom: 1rem;
    }
    .new-chat-btn button:hover { opacity: 0.92; transform: translateY(-1px); }

    .footer-note {
        font-size: 0.66rem; color: var(--text-muted); text-align: center;
        padding-top: 10px; margin-top: 6px; border-top: 1px solid var(--border);
    }

    /* Fixed-height scrollable container (native st.container(height=..)) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.sample-q-marker) {
        border-color: var(--border) !important;
        background: var(--bg-elevated) !important;
        border-radius: 10px !important;
    }

    /* ── Top bar ─────────────────────────────────────────── */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 0 14px 0; margin-bottom: 6px; border-bottom: 1px solid var(--border);
    }
    .topbar .topbar-brand { display: flex; align-items: center; gap: 9px; }
    .topbar .topbar-title { font-weight: 700; font-size: 1.05rem; color: var(--text-primary); }
    .topbar .topbar-badge {
        font-size: 0.62rem; font-weight: 600; color: var(--text-accent);
        background: rgba(0,212,255,0.10); border: 1px solid var(--border-accent);
        border-radius: 100px; padding: 1px 7px; margin-left: 4px; vertical-align: middle;
    }

    /* Theme toggle + generic top icon buttons */
    .topbar-actions .stButton > button {
        background: var(--bg-card) !important; color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important; border-radius: 9px !important;
        padding: 0.35rem 0.7rem !important; font-weight: 500 !important;
        box-shadow: none !important;
    }
    .topbar-actions .stButton > button:hover {
        border-color: var(--border-accent) !important; color: var(--text-accent) !important;
        background: var(--bg-hover) !important;
    }

    /* ── Welcome screen ──────────────────────────────────── */
    .welcome-wrap { text-align: center; padding: 3rem 1rem 1.2rem; }
    .welcome-icon { font-size: 3rem; filter: drop-shadow(0 0 22px rgba(0,212,255,.30)); }
    .welcome-title {
        margin-top: 10px; font-size: 1.7rem; font-weight: 700;
        background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .welcome-desc {
        margin: 8px auto 0; max-width: 520px; font-size: 0.92rem;
        color: var(--text-secondary); line-height: 1.65;
    }
    .welcome-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 16px; text-align: left; height: 100%;
        transition: all .22s;
    }
    .welcome-card:hover { border-color: var(--border-accent); transform: translateY(-2px); box-shadow: 0 8px 24px var(--shadow-color); }
    .welcome-card .wc-icon { font-size: 1.3rem; margin-bottom: 6px; }
    .welcome-card h4 { font-size: 0.85rem; font-weight: 600; margin-bottom: 3px; color: var(--text-primary); }
    .welcome-card p { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; margin: 0; }

    /* ── Chat messages ───────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important; padding: 6px 0 !important;
        animation: fadeInUp .25s ease;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: var(--grad) !important; color: #fff !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--bg-card) !important; border: 1px solid var(--border-accent) !important;
    }
    /* Role is identified via an explicit hidden marker (custom emoji avatars don't
       reliably expose role-specific testids), not via the avatar element itself. */
    [data-testid="stChatMessage"]:has(.msg-role-user) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has(.msg-role-user) [data-testid="stChatMessageContent"] {
        background: var(--grad) !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 11px 15px !important; max-width: 78%; margin-left: auto;
    }
    [data-testid="stChatMessage"]:has(.msg-role-user) [data-testid="stChatMessageContent"] * {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 4px 16px 16px 16px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] * {
        color: var(--text-primary) !important;
    }
    [data-testid="stChatMessageContent"] p { font-size: 0.92rem; line-height: 1.75; margin-bottom: 0.5em; }
    [data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol { padding-left: 22px; margin-bottom: 6px; }
    [data-testid="stChatMessageContent"] li { margin-bottom: 3px; }
    /* Scoped to assistant role + tag name so specificity beats the broad "*" color rule above */
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] h3 {
        color: var(--text-accent) !important; font-size: 1rem; margin: 10px 0 4px;
    }
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageContent"] strong {
        color: var(--text-accent) !important;
    }
    [data-testid="stChatMessageContent"] code {
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
        background: var(--code-bg); border: 1px solid var(--border-accent);
        border-radius: 4px; padding: 1px 5px;
    }
    [data-testid="stChatMessageContent"] pre {
        background: var(--code-bg) !important; border: 1px solid var(--border) !important;
        border-radius: 9px !important; padding: 11px !important;
    }
    [data-testid="stChatMessageContent"] pre code { background: transparent !important; border: none !important; padding: 0 !important; }
    [data-testid="stChatMessageContent"] blockquote {
        border-left: 3px solid var(--grad-start); padding-left: 10px; margin: 6px 0;
        color: var(--text-secondary); font-style: italic;
    }
    [data-testid="stChatMessageContent"] table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.8rem; }
    [data-testid="stChatMessageContent"] th {
        background: rgba(0,212,255,0.10); color: var(--text-accent);
        padding: 6px 10px; text-align: left; border: 1px solid var(--border);
    }
    [data-testid="stChatMessageContent"] td { padding: 5px 10px; border: 1px solid var(--border); color: var(--text-secondary); }

    /* Sources button under assistant messages */
    .sources-row .stButton > button, .sources-row .stPopover > div > button {
        font-size: 0.72rem !important; font-weight: 500 !important;
        color: var(--text-accent) !important; background: rgba(0,212,255,0.08) !important;
        border: 1px solid var(--border-accent) !important; border-radius: 100px !important;
        padding: 2px 12px !important; box-shadow: none !important;
    }
    [data-testid="stPopoverBody"] {
        background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        max-height: 420px !important; overflow-y: auto !important; padding-right: 6px !important;
    }
    /* Force every text node inside the Sources popover to use the theme's light
       colors — Streamlit's default markdown/caption color is dark and unreadable
       against our dark popover background. */
    [data-testid="stPopoverBody"] * { color: var(--text-primary) !important; }
    [data-testid="stPopoverBody"] p { color: var(--text-primary) !important; font-size: 0.85rem; line-height: 1.6; }
    [data-testid="stPopoverBody"] strong { color: #ffffff !important; font-size: 0.92rem; }
    [data-testid="stPopoverBody"] em { color: var(--text-secondary) !important; font-style: italic; }
    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"],
    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p,
    [data-testid="stPopoverBody"] small {
        color: var(--text-secondary) !important; font-size: 0.75rem !important;
    }
    [data-testid="stPopoverBody"] hr { border-color: var(--border) !important; margin: 10px 0 !important; }

    /* Typing / thinking spinner */
    [data-testid="stSpinner"] > div { color: var(--text-secondary) !important; }

    /* ── Chat input ──────────────────────────────────────── */
    [data-testid="stChatInput"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 24px !important; box-shadow: 0 4px 20px var(--shadow-color) !important;
        max-width: 900px; margin: 0 auto;
    }
    /* The input is wrapped in several baseweb divs that carry their own (light) background;
       force every layer transparent so our container background shows through. */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: transparent !important; border: none !important; box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        color: var(--text-primary) !important;
        background: transparent !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        font-size: 0.9rem !important;
        caret-color: var(--text-primary) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; opacity: 1 !important; }
    [data-testid="stChatInput"] button { background: var(--grad) !important; border-radius: 50% !important; border: none !important; }
    [data-testid="stChatInput"] button svg { color: #ffffff !important; fill: #ffffff !important; }
    [data-testid="stBottomBlockContainer"] { background: var(--bg-base) !important; }
    [data-testid="stBottom"] { background: var(--bg-base) !important; }
    [data-testid="stBottom"] > div { background: transparent !important; }

    /* ── Generic components ──────────────────────────────── */
    .stButton > button {
        border-radius: 9px; font-weight: 500; transition: all .2s;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--bg-card) !important; border: 1.5px dashed var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--border-accent) !important; }
    div[data-testid="stTextInput"] input {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        color: var(--text-primary) !important; border-radius: 9px !important;
    }
    div[data-testid="stTextInput"] input:focus { border-color: var(--border-accent) !important; }
    div[data-baseweb="select"] > div {
        background: var(--bg-card) !important; border-color: var(--border) !important;
        border-radius: 9px !important; color: var(--text-primary) !important;
    }
    div[data-testid="stAlert"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; color: var(--text-primary) !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] summary { color: var(--text-primary) !important; }
    hr { border-color: var(--border) !important; }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    /* ── Responsive ──────────────────────────────────────── */
    @media (max-width: 900px) {
        .block-container { max-width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { max-width: 90%; }
        .welcome-title { font-size: 1.35rem; }
        .welcome-desc { font-size: 0.85rem; }
    }
    """

    st.markdown(f"<style>{themed}{static}</style>", unsafe_allow_html=True)


inject_theme_css()

# ─── Helper Functions ──────────────────────────────────────────────────────────
def clear_chat():
    st.session_state.messages = []
    st.session_state.show_sources = False
    st.session_state.last_answer = None
    st.session_state.last_sources = []
    st.session_state.queued_question = None
    st.rerun()


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()


def refresh_knowledge_base():
    docs = get_documents()
    st.session_state.knowledge_base = pd.DataFrame(docs)


def stream_answer(placeholder, text: str):
    """Reveal the already-generated answer progressively (typing effect),
    without touching how the answer itself is produced."""
    words = text.split(" ")
    chunk_size = max(1, len(words) // 120)
    shown = ""
    for i in range(0, len(words), chunk_size):
        shown += (" " if shown else "") + " ".join(words[i:i + chunk_size])
        placeholder.markdown(shown)
        time.sleep(0.015)
    placeholder.markdown(text)


def render_sources_popover(sources, key):
    """Render a small '📚 Sources' popover scoped to a single message."""
    if not sources:
        return
    num_src = len(sources)
    st.markdown('<div class="sources-row">', unsafe_allow_html=True)
    with st.popover(f"📚 Sources ({num_src})"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** {src.get('title', 'Unknown')}")
            if src.get('page_number'):
                st.caption(f"Page {src['page_number']}")
            st.markdown(f"*{src.get('chunk_text', '')[:300]}...*")
            if i < num_src:
                st.divider()
    st.markdown('</div>', unsafe_allow_html=True)


def render_welcome_screen():
    st.markdown(f"""
    <div class="welcome-wrap">
        <div class="welcome-icon">⚛️</div>
        <div class="welcome-title">Nuclear Law Assistant</div>
        <p class="welcome-desc">
            Your AI expert on nuclear law — treaties, safeguards agreements, and liability frameworks,
            grounded in official reference documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(WELCOME_CARDS):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="welcome-card">
                <div class="wc-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">⚛️</div>
        <div>
            <div class="logo-title">Nuclear Law</div>
            <div class="logo-subtitle">Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat — most prominent action, right under the logo
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋ New Chat", use_container_width=True, key="new_chat_sidebar"):
        clear_chat()
    st.markdown('</div>', unsafe_allow_html=True)

    # Status
    doc_count = len(st.session_state.knowledge_base)
    dot_class = "status-dot" if doc_count > 0 else "status-dot empty"
    status_title = "Knowledge Base Ready" if doc_count > 0 else "Knowledge Base Empty"
    st.markdown(f"""
    <div class="status-badge">
        <span class="{dot_class}"></span>
        <div>
            <div class="status-title">{status_title}</div>
            <div class="status-sub">{doc_count} document{'s' if doc_count != 1 else ''} indexed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sidebar-section"><div class="section-title">📄 Add Documents</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None and not st.session_state.processing:
        st.session_state.processing = True
        DOCS_FOLDER = "./documents"
        if not os.path.exists(DOCS_FOLDER):
            os.makedirs(DOCS_FOLDER)
        try:
            upload_name = safe_pdf_filename(uploaded_file.name)
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.processing = False
            st.stop()
        file_path = os.path.join(DOCS_FOLDER, upload_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            doc = load_pdf(file_path)
            if doc:
                num_chunks = add_document_to_store(doc)
                refresh_knowledge_base()
                st.success(f"✅ Added {uploaded_file.name} ({num_chunks} chunks)")
            else:
                st.error("❌ Failed to extract text from PDF")
        st.session_state.processing = False
        st.rerun()

    st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
    url_input = st.text_input("Or enter PDF URL", placeholder="https://example.com/doc.pdf", label_visibility="collapsed")
    if st.button("📥 Fetch from URL", use_container_width=True, disabled=st.session_state.processing):
        if url_input.strip():
            st.session_state.processing = True
            try:
                with st.spinner("Downloading and indexing..."):
                    file_path = download_pdf_from_url(url_input.strip())
                    doc = load_pdf(file_path, source_url=url_input.strip())
                    if doc:
                        num_chunks = add_document_to_store(doc)
                        refresh_knowledge_base()
                        st.success(f"✅ Added {os.path.basename(file_path)} ({num_chunks} chunks)")
                    else:
                        st.error("❌ Failed to process the PDF")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False
            st.rerun()
        else:
            st.warning("Please enter a valid URL")

    st.markdown('</div>', unsafe_allow_html=True)

    # Knowledge Base List
    st.markdown('<div class="sidebar-section"><div class="section-title">📚 Knowledge Base</div>', unsafe_allow_html=True)
    docs_df = st.session_state.knowledge_base
    if not docs_df.empty:
        for _, row in docs_df.iterrows():
            st.markdown(f"""
            <div class="doc-item">
                <div class="doc-name">{row['title']}</div>
                <span class="doc-meta">{row.get('doc_type', 'pdf')}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No documents yet. Upload a PDF or add a URL.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="sidebar-section"><div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)
    st.selectbox("Source Type", ["All", "Official IAEA", "User Uploaded"], key="filter_source")
    if not docs_df.empty:
        st.selectbox("Document", ["All"] + list(docs_df['title'].unique()), key="filter_doc")
    else:
        st.selectbox("Document", ["All"], key="filter_doc")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sample Questions — fixed-height, independently scrollable container
    st.markdown('<div class="sidebar-section"><div class="section-title">💡 Sample Questions</div>', unsafe_allow_html=True)
    st.markdown('<span class="sample-q-marker" style="display:none"></span>', unsafe_allow_html=True)
    with st.container(height=220, border=True):
        st.markdown('<div class="sample-q-wrap">', unsafe_allow_html=True)
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
                st.session_state.queued_question = q
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer-note">For research purposes only</div>', unsafe_allow_html=True)

# ─── Main Area: Top Bar ─────────────────────────────────────────────────────────
top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown(f"""
    <div class="topbar-brand" style="display:flex;align-items:center;gap:9px;">
        {ATOM_SVG}
        <span class="topbar-title">Nuclear Law Assistant<span class="topbar-badge">AI</span></span>
    </div>
    """, unsafe_allow_html=True)
with top_right:
    st.markdown('<div class="topbar-actions">', unsafe_allow_html=True)
    theme_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    if st.button(theme_label, help="Toggle theme", use_container_width=True, key="theme_toggle_top"):
        toggle_theme()
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr style="margin: 10px 0 14px 0;">', unsafe_allow_html=True)

# ─── Chat Messages / Welcome Screen ─────────────────────────────────────────────
if not st.session_state.messages:
    render_welcome_screen()
else:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚛️"):
            st.markdown(f'<span class="msg-role-{msg["role"]}" style="display:none"></span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources_popover(msg["sources"], key=f"cite_hist_{idx}")

# ─── Input & Processing ──────────────────────────────────────────────────────
prompt = st.chat_input("Ask about nuclear law...")
question_to_process = st.session_state.pop("queued_question", None) or prompt

if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown('<span class="msg-role-user" style="display:none"></span>', unsafe_allow_html=True)
        st.markdown(question_to_process)

    with st.chat_message("assistant", avatar="⚛️"):
        st.markdown('<span class="msg-role-assistant" style="display:none"></span>', unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            context = retrieve_context(question_to_process)
            result = get_answer(question_to_process, context)
            answer = result['answer']
            sources = result['sources']

        answer_placeholder = st.empty()
        stream_answer(answer_placeholder, answer)

        if sources:
            render_sources_popover(sources, key=f"cite_new_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()
