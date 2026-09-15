"""
Conversation Memory Agent: Persists multi-turn conversation state, provides history query,
and supports transcript export/import.
"""
import json
import time
from typing import List, Dict, Optional, Deque
from collections import deque
from pathlib import Path
import logging

from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, ConversationTurn

logger = logging.getLogger(__name__)


class ConversationMemoryAgent(BaseAgent):
    """
    Conversation Memory Agent: Manages session history, turns, and serialization.
    """
    def __init__(
        self,
        max_history_turns: int = 20,
        timeout: float = 0.2,
        max_sessions: int = 100,
        session_ttl_sec: float = 1800.0
    ):
        super().__init__(name="ConversationMemoryAgent", timeout=timeout)
        self.max_history_turns = max_history_turns
        self.max_sessions = max_sessions
        self.session_ttl_sec = session_ttl_sec
        self.sessions: Dict[str, Deque[ConversationTurn]] = {}
        self.session_last_active: Dict[str, float] = {}

    def _touch_session(self, session_id: str) -> None:
        self.session_last_active[session_id] = time.time()
        if len(self.session_last_active) > self.max_sessions:
            self.cleanup_stale_sessions(self.session_ttl_sec)
            # If still over limit, drop oldest non-default session
            if len(self.session_last_active) > self.max_sessions:
                sorted_sessions = sorted(
                    [s for s in self.session_last_active if s != "default_session"],
                    key=lambda s: self.session_last_active[s]
                )
                for drop_id in sorted_sessions[:len(self.session_last_active) - self.max_sessions]:
                    self.clear_session(drop_id)

    def cleanup_stale_sessions(self, max_age_seconds: Optional[float] = None) -> int:
        """Evict sessions inactive for longer than max_age_seconds."""
        ttl = max_age_seconds if max_age_seconds is not None else self.session_ttl_sec
        now = time.time()
        stale = [
            sid for sid, last in list(self.session_last_active.items())
            if sid != "default_session" and (now - last) > ttl
        ]
        for sid in stale:
            self.clear_session(sid)
        return len(stale)

    def get_session_history(self, session_id: str) -> List[ConversationTurn]:
        """Get turns for a given session."""
        self._touch_session(session_id)
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_history_turns)
        return list(self.sessions[session_id])

    def add_turn(
        self,
        session_id: str,
        raw_signs: List[str],
        disambiguated_signs: List[str],
        english_translation: str
    ) -> ConversationTurn:
        """Record a completed conversation turn."""
        self._touch_session(session_id)
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_history_turns)

        turn_id = len(self.sessions[session_id]) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            raw_signs=raw_signs,
            disambiguated_signs=disambiguated_signs,
            english_translation=english_translation,
            timestamp=time.time()
        )
        self.sessions[session_id].append(turn)
        return turn

    def clear_session(self, session_id: str):
        """Clear turns and remove tracking for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            del self.sessions[session_id]
        if session_id in self.session_last_active:
            del self.session_last_active[session_id]

    def export_transcript_json(self, session_id: str) -> str:
        """Export session transcript to JSON."""
        history = self.get_session_history(session_id)
        turns_data = [turn.model_dump() for turn in history]
        return json.dumps({
            "session_id": session_id,
            "turn_count": len(turns_data),
            "turns": turns_data
        }, indent=2)

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Updates session history and attaches recent history to state for downstream/upstream agents.
        """
        session_id = state.session_id or "default_session"
        
        # If this state has a completed sentence and signs, commit as a turn
        if state.english_sentence and (state.disambiguated_signs or state.recognized_signs):
            raw_signs = [p.sign for p in state.recognized_signs] if state.recognized_signs else state.disambiguated_signs
            self.add_turn(
                session_id=session_id,
                raw_signs=raw_signs,
                disambiguated_signs=state.disambiguated_signs,
                english_translation=state.english_sentence
            )
            
        state.history = self.get_session_history(session_id)
        return state
