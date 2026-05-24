import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from triage_chat.agent import triage_agent

router = APIRouter()

# In-memory session store: session_id → message history
_sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # omit to start a new session


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/chat/triage", response_model=ChatResponse)
async def chat_triage(body: ChatRequest):
    if triage_agent.agent is None:
        raise HTTPException(status_code=503, detail="Triage agent not ready")

    session_id = body.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])
    history.append({"role": "user", "content": body.message})

    reply = await triage_agent.chat(history)

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    return ChatResponse(reply=reply, session_id=session_id)
