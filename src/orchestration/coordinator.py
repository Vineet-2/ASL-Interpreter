"""
Orchestrator: Async Blackboard Coordinator for the 6-agent ASL Conversation Pipeline.
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.models.schema import ConversationState, AgentStatus, AgentState, SignPrediction
from src.agents.vision_agent import VisionAgent
from src.agents.sign_agent import SignRecognitionAgent
from src.agents.context_agent import ContextAgent
from src.agents.language_agent import LanguageAgent
from src.agents.speech_agent import SpeechAgent
from src.agents.memory_agent import ConversationMemoryAgent

logger = logging.getLogger(__name__)

# Default profiles for live stabilization
SENSITIVITY_PROFILES = {
    "steady": {
        "stable_frames": 10,       # ~500ms steady hold before committing sign
        "idle_commit_cycles": 26,   # ~1.3s pause before sentence translation
        "cooldown_frames": 8,       # Debounce cooldown between tokens
    },
    "balanced": {
        "stable_frames": 8,        # ~400ms steady hold
        "idle_commit_cycles": 20,   # ~1.0s pause
        "cooldown_frames": 6,
    },
    "fast": {
        "stable_frames": 5,        # ~250ms steady hold
        "idle_commit_cycles": 14,   # ~0.7s pause
        "cooldown_frames": 4,
    }
}


@dataclass
class LiveSession:
    pending: List[SignPrediction] = field(default_factory=list)
    stable_sign: Optional[str] = None
    stable_count: int = 0
    idle_cycles: int = 0
    cooldown: int = 0
    sensitivity: str = "steady"


class Orchestrator:
    """
    Central Coordinator managing agent pipeline execution, blackboard state handoff,
    failure containment, and telemetry.
    """
    def __init__(
        self,
        vision_agent: Optional[VisionAgent] = None,
        sign_agent: Optional[SignRecognitionAgent] = None,
        context_agent: Optional[ContextAgent] = None,
        language_agent: Optional[LanguageAgent] = None,
        speech_agent: Optional[SpeechAgent] = None,
        memory_agent: Optional[ConversationMemoryAgent] = None,
        max_sessions: int = 50,
        session_ttl_sec: float = 1800.0,
    ):
        self._default_vision = vision_agent
        self.vision_agents: Dict[str, VisionAgent] = {}
        if vision_agent is not None:
            self.vision_agents["default_session"] = vision_agent
        
        self.sign = sign_agent or SignRecognitionAgent()
        self.context = context_agent or ContextAgent()
        self.language = language_agent or LanguageAgent()
        self.speech = speech_agent or SpeechAgent()
        self.memory = memory_agent or ConversationMemoryAgent()

        self.live_sessions: Dict[str, LiveSession] = {}
        self.session_last_active: Dict[str, float] = {}
        self.max_sessions = max_sessions
        self.session_ttl_sec = session_ttl_sec

    def _touch_session(self, session_id: str) -> None:
        self.session_last_active[session_id] = time.time()
        if len(self.session_last_active) > self.max_sessions:
            self.cleanup_stale_sessions(self.session_ttl_sec)
            # If still over limit, drop oldest non-default sessions
            if len(self.session_last_active) > self.max_sessions:
                sorted_sessions = sorted(
                    [s for s in self.session_last_active if s != "default_session"],
                    key=lambda s: self.session_last_active[s]
                )
                for drop_id in sorted_sessions[:len(self.session_last_active) - self.max_sessions]:
                    self.evict_session(drop_id)

    def evict_session(self, session_id: str) -> None:
        """Explicitly evicts a session and releases any associated resources."""
        if session_id in self.vision_agents:
            va = self.vision_agents.pop(session_id)
            va.reset()
            if hasattr(va, "hands") and va.hands is not None:
                try:
                    va.hands.close()
                except Exception:
                    pass
            if hasattr(va, "holistic") and va.holistic is not None:
                try:
                    va.holistic.close()
                except Exception:
                    pass
        if session_id in self.live_sessions:
            del self.live_sessions[session_id]
        if session_id in self.session_last_active:
            del self.session_last_active[session_id]
        self.memory.clear_session(session_id)

    def cleanup_stale_sessions(self, max_age_seconds: Optional[float] = None) -> int:
        """Evicts sessions inactive for longer than max_age_seconds."""
        ttl = max_age_seconds if max_age_seconds is not None else self.session_ttl_sec
        now = time.time()
        stale = [
            sid for sid, last in list(self.session_last_active.items())
            if sid != "default_session" and (now - last) > ttl
        ]
        for sid in stale:
            self.evict_session(sid)
        self.memory.cleanup_stale_sessions(ttl)
        return len(stale)

    @property
    def vision(self) -> VisionAgent:
        return self._get_vision("default_session")

    def _get_vision(self, session_id: str) -> VisionAgent:
        self._touch_session(session_id)
        if session_id not in self.vision_agents:
            if self._default_vision is not None and not self.vision_agents:
                self.vision_agents[session_id] = self._default_vision
            else:
                self.vision_agents[session_id] = VisionAgent()
        return self.vision_agents[session_id]

    @property
    def agents(self) -> List[Any]:
        return [
            self.vision,
            self.sign,
            self.context,
            self.language,
            self.speech,
            self.memory
        ]

    def _session(self, session_id: str) -> LiveSession:
        self._touch_session(session_id)
        if session_id not in self.live_sessions:
            self.live_sessions[session_id] = LiveSession()
        return self.live_sessions[session_id]

    def reset_live_session(self, session_id: str) -> None:
        self._touch_session(session_id)
        self.live_sessions[session_id] = LiveSession()
        if session_id in self.vision_agents:
            self.vision_agents[session_id].reset()

    def set_session_sensitivity(self, session_id: str, sensitivity: str) -> None:
        if sensitivity in SENSITIVITY_PROFILES:
            self._session(session_id).sensitivity = sensitivity

    def _update_live_tokens(
        self,
        session_id: str,
        hypotheses: List[SignPrediction],
        hands_present: bool,
        force_commit: bool,
    ) -> Tuple[bool, float]:
        """
        Stabilize flickering classifier output into a phrase, then commit on pause.
        Returns (should_commit, hold_progress 0.0-1.0).
        """
        live = self._session(session_id)
        profile = SENSITIVITY_PROFILES.get(live.sensitivity, SENSITIVITY_PROFILES["steady"])
        stable_req = profile["stable_frames"]
        idle_req = profile["idle_commit_cycles"]
        cooldown_req = profile["cooldown_frames"]

        # Handle refractory cooldown
        if live.cooldown > 0:
            live.cooldown -= 1
            live.stable_count = 0
            live.stable_sign = None
            return False, 0.0

        hypothesis = hypotheses[0].sign if hypotheses and hypotheses[0].sign else None
        hold_progress = 0.0

        if hypothesis and hands_present:
            live.idle_cycles = 0
            if hypothesis == live.stable_sign:
                live.stable_count += 1
            else:
                live.stable_sign = hypothesis
                live.stable_count = 1

            hold_progress = min(1.0, live.stable_count / float(stable_req))
            last = live.pending[-1].sign if live.pending else None

            if live.stable_count >= stable_req and hypothesis != last:
                live.pending.append(hypotheses[0])
                live.cooldown = cooldown_req
                live.stable_sign = None
                live.stable_count = 0
                hold_progress = 1.0
        else:
            live.stable_sign = None
            live.stable_count = 0
            hold_progress = 0.0
            if not hands_present:
                live.idle_cycles += 1
            else:
                live.idle_cycles = 0

        should_commit = False
        if live.pending and (force_commit or live.idle_cycles >= idle_req):
            should_commit = True
        return should_commit, round(hold_progress, 2)

    async def run_pipeline(
        self,
        state: ConversationState,
        commit_phrase: bool = False,
        hands_present: bool = True,
        accumulate_live: bool = False,
    ) -> ConversationState:
        """
        Executes the 6-agent sequential blackboard pipeline with per-agent timeout isolation.
        """
        pipeline_start = time.perf_counter()
        manual_glosses = bool(state.disambiguated_signs) and not state.raw_landmarks

        if not manual_glosses:
            # Step 1: Vision Agent (Session-isolated)
            vision_agent = self._get_vision(state.session_id)
            try:
                state = await asyncio.wait_for(vision_agent.run(state), timeout=vision_agent.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(vision_agent.name, AgentState.ERROR, vision_agent.timeout * 1000, "Timeout exceeded")

            # Step 2: Sign Recognition Agent
            try:
                state = await asyncio.wait_for(self.sign.run(state), timeout=self.sign.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(self.sign.name, AgentState.ERROR, self.sign.timeout * 1000, "Timeout exceeded")
        else:
            state.update_agent_status("VisionAgent", AgentState.SKIPPED, 0.0, "Manual gloss input")
            state.update_agent_status(self.sign.name, AgentState.SKIPPED, 0.0, "Manual gloss input")

        live = self._session(state.session_id)
        live_hypothesis = [p.sign for p in state.recognized_signs]
        live_conf = state.recognized_signs[0].confidence if state.recognized_signs else 0.0

        # Live camera path: accumulate glosses, translate only after a pause or explicit commit.
        # Testing-lab / REST path still translates immediately when signs are provided.
        if accumulate_live:
            should_commit, hold_progress = self._update_live_tokens(
                state.session_id,
                state.recognized_signs,
                hands_present=hands_present,
                force_commit=commit_phrase,
            )
            state.metadata["pending_glosses"] = [p.sign for p in live.pending]
            state.metadata["live_hypothesis"] = live_hypothesis[0] if live_hypothesis else None
            state.metadata["live_confidence"] = round(float(live_conf), 3)
            state.metadata["hold_progress"] = hold_progress
            state.metadata["hands_present"] = hands_present
            state.metadata["sign_backend"] = state.metadata.get("sign_backend") or getattr(self.sign, "backend", "unknown")

            if should_commit:
                state.recognized_signs = list(live.pending)
                live.pending = []
                live.idle_cycles = 0
                live.stable_sign = None
                live.stable_count = 0
                live.cooldown = 0
            else:
                for agent in [self.context, self.language, self.speech, self.memory]:
                    state.update_agent_status(agent.name, AgentState.IDLE, 0.0)
                total_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0
                state.total_pipeline_latency_ms = round(total_latency_ms, 2)
                return state

        has_signs = bool(state.recognized_signs or state.disambiguated_signs)

        if has_signs:
            # Step 3: Populate history from Memory Agent prior to Context disambiguation
            state.history = self.memory.get_session_history(state.session_id)

            # Step 4: Context Agent (Disambiguation)
            try:
                state = await asyncio.wait_for(self.context.run(state), timeout=self.context.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(self.context.name, AgentState.ERROR, self.context.timeout * 1000, "Timeout exceeded")
                # Fallback to raw signs
                state.disambiguated_signs = [p.sign for p in state.recognized_signs]

            # Step 5: Language Agent (Grammar Translation)
            try:
                state = await asyncio.wait_for(self.language.run(state), timeout=self.language.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(self.language.name, AgentState.DEGRADED, self.language.timeout * 1000, "Language API timeout -> local fallback")
                state.english_sentence = self.language.translate_with_local_rules(state.disambiguated_signs)
                state.translation_provider = "timeout_fallback"

            # Step 6: Speech Agent (TTS)
            try:
                state = await asyncio.wait_for(self.speech.run(state), timeout=self.speech.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(self.speech.name, AgentState.DEGRADED, self.speech.timeout * 1000, "Speech synthesis timeout")

            # Step 7: Memory Agent (Commit completed turn)
            try:
                state = await asyncio.wait_for(self.memory.run(state), timeout=self.memory.timeout)
            except asyncio.TimeoutError:
                state.update_agent_status(self.memory.name, AgentState.ERROR, self.memory.timeout * 1000, "Memory commit timeout")
        else:
            # No signs detected in this cycle - mark downstream agents idle/skipped
            for agent in [self.context, self.language, self.speech, self.memory]:
                state.update_agent_status(agent.name, AgentState.IDLE, 0.0)

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0
        state.total_pipeline_latency_ms = round(total_latency_ms, 2)
        return state

    def get_all_agent_statuses(self) -> Dict[str, AgentStatus]:
        """Returns the current operational status of all 6 agents."""
        return {agent.name: agent.get_status() for agent in self.agents}
