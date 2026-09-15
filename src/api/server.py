"""
FastAPI Backend Server and WebSocket Streaming API for ASL Assistant.
"""
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.schema import ConversationState, AgentState, AgentStatus, ConversationTurn
from src.orchestration.coordinator import Orchestrator
from src.config import BASE_DIR, HOST, PORT, MODELS_DIR

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic ASL Conversation Assistant API",
    description="Real-time multi-agent American Sign Language conversation system.",
    version="1.0.0"
)

# Enable CORS for local dev / frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Orchestrator instance
orchestrator = Orchestrator()


class PredictRequest(BaseModel):
    session_id: str = "default_session"
    raw_landmarks: Optional[List[List[float]]] = None
    signs: Optional[List[str]] = None


class TranslateRequest(BaseModel):
    session_id: str = "default_session"
    signs: List[str]


class AmbiguityBenchmarkCase(BaseModel):
    id: int
    raw_signs: List[str]
    context_turn: Optional[str]
    expected_disambiguated: List[str]
    description: str


# 16 standard benchmark test cases across homonyms, pronouns, verbs, and social terms
AMBIGUITY_BENCHMARK_CASES = [
    {
        "id": 1,
        "raw_signs": ["ME", "WANT", "EAT", "NOW"],
        "history_context": None,
        "expected": ["I", "WANT", "FOOD", "NOW"],
        "description": "EAT nominalized as FOOD following verb WANT"
    },
    {
        "id": 2,
        "raw_signs": ["ME", "GO", "EAT"],
        "history_context": None,
        "expected": ["I", "GO", "EAT"],
        "description": "EAT as infinitive action verb following GO"
    },
    {
        "id": 3,
        "raw_signs": ["ME", "GO", "SHOP", "TODAY"],
        "history_context": None,
        "expected": ["I", "GO", "STORE", "TODAY"],
        "description": "SHOP disambiguated to STORE (location destination) following GO"
    },
    {
        "id": 4,
        "raw_signs": ["ME", "LIKE", "SHOP", "LATER"],
        "history_context": None,
        "expected": ["I", "LIKE", "SHOP", "LATER"],
        "description": "SHOP preserved as action verb following LIKE + time adverb"
    },
    {
        "id": 5,
        "raw_signs": ["YOU", "HELP", "ME", "PLEASE"],
        "history_context": None,
        "expected": ["YOU", "HELP", "ME", "PLEASE"],
        "description": "ME preserved in objective case after verb HELP"
    },
    {
        "id": 6,
        "raw_signs": ["ME", "PHONE", "WHERE"],
        "history_context": None,
        "expected": ["MY", "PHONE", "WHERE"],
        "description": "ME resolved to possessive MY preceding noun PHONE"
    },
    {
        "id": 7,
        "raw_signs": ["ME", "HOME", "GO", "NOW"],
        "history_context": None,
        "expected": ["MY", "HOME", "GO", "NOW"],
        "description": "ME resolved to possessive MY before noun HOME"
    },
    {
        "id": 8,
        "raw_signs": ["YOU", "CAR", "WHERE"],
        "history_context": None,
        "expected": ["YOUR", "CAR", "WHERE"],
        "description": "YOU resolved to possessive YOUR before noun CAR"
    },
    {
        "id": 9,
        "raw_signs": ["ME", "WANT", "DRINK"],
        "history_context": None,
        "expected": ["I", "WANT", "BEVERAGE"],
        "description": "DRINK nominalized to beverage noun after WANT"
    },
    {
        "id": 10,
        "raw_signs": ["WE", "DRINK", "WATER"],
        "history_context": None,
        "expected": ["WE", "DRINK", "WATER"],
        "description": "DRINK as transitive action verb before WATER"
    },
    {
        "id": 11,
        "raw_signs": ["ME", "FINISH", "WORK"],
        "history_context": None,
        "expected": ["I", "FINISH", "WORKPLACE"],
        "description": "WORK resolved as noun/workplace after FINISH"
    },
    {
        "id": 12,
        "raw_signs": ["ME", "WORK", "TODAY"],
        "history_context": None,
        "expected": ["I", "WORK", "TODAY"],
        "description": "WORK resolved as verb before TODAY"
    },
    {
        "id": 13,
        "raw_signs": ["HE_SHE", "TIRED"],
        "history_context": "My sister came to visit yesterday.",
        "expected": ["SHE", "TIRED"],
        "description": "HE_SHE resolved to SHE from prior turn context referencing sister"
    },
    {
        "id": 14,
        "raw_signs": ["HE_SHE", "GO", "SCHOOL"],
        "history_context": "My brother is preparing for class.",
        "expected": ["HE", "GO", "SCHOOL"],
        "description": "HE_SHE resolved to HE from prior turn context referencing brother"
    },
    {
        "id": 15,
        "raw_signs": ["THANKYOU", "ME", "NEED", "HELP"],
        "history_context": None,
        "expected": ["THANKYOU", "I", "NEED", "HELP"],
        "description": "Social greeting prefix with ME nominative subject before NEED"
    },
    {
        "id": 16,
        "raw_signs": ["ME", "MORE", "EAT", "PLEASE"],
        "history_context": None,
        "expected": ["I", "MORE", "FOOD", "PLEASE"],
        "description": "EAT modified by MORE resolved to FOOD noun"
    }
]


@app.get("/api/agents/status")
async def get_agents_status():
    """Return status, health, and latency for all 6 agents."""
    return {
        "status": "healthy",
        "agents": orchestrator.get_all_agent_statuses()
    }


@app.get("/api/health")
async def health():
    """Runtime recognition backend (trained weights vs geometric fallback)."""
    return {
        "status": "healthy",
        "sign_backend": getattr(orchestrator.sign, "backend", "unknown"),
        "weights_loaded": bool(getattr(orchestrator.sign, "weights_loaded", False)),
    }


@app.post("/api/session/reset")
async def reset_session(session_id: str = Query("default_session")):
    """Reset a conversation session and clear memory history."""
    orchestrator.memory.clear_session(session_id)
    orchestrator.reset_live_session(session_id)
    return {"status": "success", "message": f"Session '{session_id}' cleared"}


@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """Retrieve full transcript history for a session."""
    raw_json = orchestrator.memory.export_transcript_json(session_id)
    return JSONResponse(content=json.loads(raw_json))


@app.post("/api/pipeline/run")
async def run_pipeline_endpoint(req: PredictRequest):
    """
    Direct endpoint to run the full 6-agent pipeline.
    """
    state = ConversationState(session_id=req.session_id)
    
    if req.raw_landmarks:
        state.raw_landmarks = req.raw_landmarks
    elif req.signs:
        state.disambiguated_signs = req.signs

    result_state = await orchestrator.run_pipeline(state)
    return result_state.model_dump()


@app.post("/api/translate")
async def translate_signs(req: TranslateRequest):
    """
    Directly run Context -> Language -> Speech -> Memory chain on a sign sequence.
    """
    state = ConversationState(
        session_id=req.session_id,
        disambiguated_signs=req.signs
    )
    result_state = await orchestrator.run_pipeline(state)
    return {
        "raw_signs": req.signs,
        "disambiguated_signs": result_state.disambiguated_signs,
        "english_sentence": result_state.english_sentence,
        "provider": result_state.translation_provider,
        "audio_base64": result_state.speech_audio_base64,
        "latency_ms": result_state.total_pipeline_latency_ms,
        "agent_statuses": result_state.agent_statuses
    }


@app.get("/api/benchmark/ambiguity")
async def run_ambiguity_benchmark():
    """
    Evaluates Context Agent against the 16 standard benchmark test cases vs. No-Context baseline.
    """
    results = []
    context_correct = 0
    no_context_correct = 0
    total = len(AMBIGUITY_BENCHMARK_CASES)

    for case in AMBIGUITY_BENCHMARK_CASES:
        raw = case["raw_signs"]
        expected = case["expected"]
        history_text = case.get("history_context")
        
        # Build mock history turn if case specifies context
        history = []
        if history_text:
            history = [ConversationTurn(
                turn_id=1,
                raw_signs=[],
                disambiguated_signs=[],
                english_translation=history_text
            )]

        # Context Agent resolution
        resolved_with_context, notes = orchestrator.context.disambiguate_sequence(raw, history)
        is_context_correct = (resolved_with_context == expected)
        if is_context_correct:
            context_correct += 1

        # No-Context baseline (raw unmapped signs with naive uppercase)
        no_context = [s.upper() for s in raw]
        is_no_context_correct = (no_context == expected)
        if is_no_context_correct:
            no_context_correct += 1

        results.append({
            "case_id": case["id"],
            "description": case["description"],
            "raw_signs": raw,
            "expected": expected,
            "with_context": resolved_with_context,
            "without_context": no_context,
            "context_correct": is_context_correct,
            "no_context_correct": is_no_context_correct,
            "notes": notes
        })

    context_acc = round((context_correct / total) * 100, 1)
    no_context_acc = round((no_context_correct / total) * 100, 1)

    return {
        "total_cases": total,
        "context_agent_accuracy_pct": context_acc,
        "no_context_accuracy_pct": no_context_acc,
        "improvement_pct": round(context_acc - no_context_acc, 1),
        "results": results
    }


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for bi-directional real-time landmark/sign streaming and telemetry.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or "default_session"
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "landmarks")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue

            if msg_type == "reset":
                orchestrator.memory.clear_session(session_id)
                orchestrator.reset_live_session(session_id)
                orchestrator._get_vision(session_id).buffer.clear()
                await websocket.send_json({
                    "type": "pipeline_result",
                    "session_id": session_id,
                    "recognized_signs": [],
                    "disambiguated_signs": [],
                    "english_sentence": None,
                    "pending_glosses": [],
                    "live_hypothesis": None,
                    "sign_backend": getattr(orchestrator.sign, "backend", "unknown"),
                })
                continue
                
            if "sensitivity" in data:
                orchestrator.set_session_sensitivity(session_id, data["sensitivity"])

            state = ConversationState(session_id=session_id)
            
            commit_phrase = bool(data.get("commit")) or msg_type == "commit"
            hands_present = bool(data.get("hands_present", True))
            accumulate_live = False

            if msg_type == "signs":
                # Direct sign simulation input
                signs = data.get("signs", [])
                state.disambiguated_signs = signs
            elif msg_type in ("landmarks", "commit"):
                landmarks = data.get("landmarks", [])
                if landmarks:
                    state.raw_landmarks = landmarks
                accumulate_live = True
                if msg_type == "commit":
                    commit_phrase = True
            
            # Run blackboard pipeline
            result_state = await orchestrator.run_pipeline(
                state,
                commit_phrase=commit_phrase,
                hands_present=hands_present,
                accumulate_live=accumulate_live,
            )
            
            # Send live telemetry and response back to UI
            response_payload = {
                "type": "pipeline_result",
                "session_id": session_id,
                "recognized_signs": [p.model_dump() for p in result_state.recognized_signs],
                "disambiguated_signs": result_state.disambiguated_signs,
                "context_notes": result_state.context_notes,
                "english_sentence": result_state.english_sentence,
                "translation_provider": result_state.translation_provider,
                "audio_base64": result_state.speech_audio_base64,
                "total_latency_ms": result_state.total_pipeline_latency_ms,
                "agent_statuses": {k: v.model_dump() for k, v in result_state.agent_statuses.items()},
                "history_turn_count": len(result_state.history),
                "pending_glosses": result_state.metadata.get("pending_glosses", []),
                "live_hypothesis": result_state.metadata.get("live_hypothesis"),
                "live_confidence": result_state.metadata.get("live_confidence", 0),
                "hold_progress": result_state.metadata.get("hold_progress", 0.0),
                "sensitivity": orchestrator._session(session_id).sensitivity,
                "hands_present": result_state.metadata.get("hands_present", hands_present),
                "sign_backend": result_state.metadata.get("sign_backend") or getattr(orchestrator.sign, "backend", "unknown"),
                "buffer_len": result_state.metadata.get("vision_buffer_len", 0),
            }
            await websocket.send_json(response_payload)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# Mount static directory for frontend
static_dir = BASE_DIR / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Agentic ASL Assistant API is running! Static UI not yet deployed.</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host=HOST, port=PORT, reload=True)
