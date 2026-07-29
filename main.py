"""
Drafter Agent - FastAPI backend

Behavior contract (enforced via system prompt + tool wiring):
- "write/update/draft ..."      -> calls update_text only. Document is shown, nothing is sent or saved.
- "save it / save as X"         -> calls save_content only. Nothing is emailed.
- "send it to X" / "email it"   -> calls save_content THEN send_email (both), in that order.
- The current document content is always returned to the UI after every turn.

Run:
    export GROQ_API_KEY=...        # required
    export EMAIL_ADDRESS=...       # your gmail address
    export EMAIL_APP_PASSWORD=...  # a Gmail App Password (NOT your normal password)
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Annotated, Sequence, TypedDict, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# --------------------------------------------------------------------------
# Config / secrets - read from environment, never hardcoded.
# --------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_PASSWORD")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

# --------------------------------------------------------------------------
# In-memory session store.
# Keyed by session_id so multiple browser tabs/users don't collide.
# For a real multi-user deployment, swap this for Redis / a DB.
# --------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.document_content: str = ""
        self.messages: List[BaseMessage] = []


SESSIONS: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session()
    return SESSIONS[session_id]


# --------------------------------------------------------------------------
# Tools
# NOTE: LangChain tools are module-level, so they can't close over a
# per-session object directly. We use a small "current session" pointer
# that the graph invocation sets before calling the LLM/tools, and clears
# after. This keeps the tool signatures untouched (as in the original code)
# while still supporting multiple sessions safely (single-threaded per call).
# --------------------------------------------------------------------------
_active_session: Session | None = None


@tool
def update_text(content: str) -> str:
    """Updates the document with the provided content."""
    global _active_session
    _active_session.document_content = content
    return f"Document has been updated successfully!\nThe current content is:\n{_active_session.document_content}"


@tool
def save_content(file_name: str) -> str:
    """Saves the current document to a text file.

    Args:
        file_name: Name for the text file (with or without .txt extension).
    """
    global _active_session
    if not file_name.endswith(".txt"):
        file_name = f"{file_name}.txt"
    try:
        os.makedirs("saved_documents", exist_ok=True)
        path = os.path.join("saved_documents", file_name)
        with open(path, "w") as f:
            f.write(_active_session.document_content)
        return f"Document has been saved to {path}"
    except Exception as e:
        return f"Error in saving the file: {str(e)}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to the given address with the given subject and body."""
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return (
            "Error: email is not configured on the server. "
            "Set EMAIL_ADDRESS and EMAIL_APP_PASSWORD environment variables."
        )
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return f"Email successfully sent to {to}"
    except Exception as e:
        return f"Error in sending email: {e}"


tools = [update_text, save_content, send_email]
tools_by_name = {t.name: t for t in tools}

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7,
).bind_tools(tools)


# --------------------------------------------------------------------------
# LangGraph state + nodes
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are a helpful email-writing assistant. You help the user draft, update, save, \
and send an email/document. The current document content is shown below.

Rules you MUST follow:
- If the user asks you to write, draft, update, or change the content, call the 'update_text' tool \
with the FULL updated content (not just the delta), then briefly confirm and show the new content.
- If the user asks you to save (and does NOT mention sending), call 'save_content' only. Do NOT send an email.
- If the user asks you to send/email the document, call 'save_content' first, then call 'send_email'. \
Both tools must be called in that order.
- Never call 'send_email' unless the user has explicitly asked to send/email something.
- Never call 'save_content' unless the user has explicitly asked to save or send.
- After any tool call, summarize what happened and show the current document content.

Current document content:
{document_content}
"""


def agent_node(state: AgentState) -> AgentState:
    system_prompt = SYSTEM_PROMPT.format(document_content=_active_session.document_content or "(empty)")
    all_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(all_messages)
    return {"messages": [response]}


def tools_condition(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("Agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("Agent")
graph.add_conditional_edges("Agent", tools_condition, {"tools": "tools", "end": END})
graph.add_edge("tools", "Agent")
agent = graph.compile()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(title="Drafter Agent")


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    reply: str
    document: str
    tool_events: List[str] = []


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _active_session
    session = get_session(req.session_id)
    _active_session = session  # point tools at this session for the duration of the call

    session.messages.append(HumanMessage(content=req.message))

    try:
        result = agent.invoke({"messages": session.messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _active_session = None

    session.messages = list(result["messages"])

    # Collect any tool results that happened this turn, for transparency in the UI
    tool_events = [m.content for m in session.messages if isinstance(m, ToolMessage)][-5:]

    # Find the last plain AI text reply (the final, non-tool-call message)
    reply = ""
    for m in reversed(session.messages):
        if isinstance(m, AIMessage) and m.content:
            reply = m.content
            break

    return ChatResponse(reply=reply, document=session.document_content, tool_events=tool_events)


@app.get("/document")
def get_document(session_id: str = "default"):
    session = get_session(session_id)
    return {"document": session.document_content}


@app.post("/reset")
def reset(session_id: str = "default"):
    SESSIONS[session_id] = Session()
    return {"status": "reset"}


# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
