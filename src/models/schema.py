"""
Pydantic schemas and contracts for ConversationState, AgentStatus, and message payloads.
"""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import time


class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class AgentStatus(BaseModel):
    agent_name: str
    state: AgentState = AgentState.IDLE
    latency_ms: float = 0.0
    message: Optional[str] = None
    last_updated: float = Field(default_factory=time.time)


class SignPrediction(BaseModel):
    sign: str
    confidence: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    timestamp: float = Field(default_factory=time.time)


class ConversationTurn(BaseModel):
    turn_id: int
    raw_signs: List[str]
    disambiguated_signs: List[str]
    english_translation: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AmbiguityRule(BaseModel):
    trigger_sign: str
    context_keywords: List[str]
    resolved_sign: str
    description: str


class ConversationState(BaseModel):
    session_id: str = "default_session"
    timestamp: float = Field(default_factory=time.time)
    
    # Vision & Sign Recognition outputs
    raw_landmarks: Optional[List[List[float]]] = None
    landmark_shape: Optional[List[int]] = None
    recognized_signs: List[SignPrediction] = Field(default_factory=list)
    
    # Context Agent output
    disambiguated_signs: List[str] = Field(default_factory=list)
    context_notes: List[str] = Field(default_factory=list)
    
    # Language Agent output
    english_sentence: Optional[str] = None
    translation_provider: Optional[str] = None  # e.g., "groq", "gemini", "local_rules"
    
    # Speech Agent output
    speech_audio_base64: Optional[str] = None
    speech_duration_sec: Optional[float] = None
    tts_format: str = "wav"
    
    # Orchestrator & Telemetry
    agent_statuses: Dict[str, AgentStatus] = Field(default_factory=dict)
    total_pipeline_latency_ms: float = 0.0
    
    # Memory
    history: List[ConversationTurn] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_agent_status(self, name: str, state: AgentState, latency_ms: float = 0.0, msg: Optional[str] = None):
        self.agent_statuses[name] = AgentStatus(
            agent_name=name,
            state=state,
            latency_ms=latency_ms,
            message=msg,
            last_updated=time.time()
        )
