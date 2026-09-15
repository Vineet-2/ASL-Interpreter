"""
Unit tests for ConversationState and schema objects.
"""
import pytest
from src.models.schema import ConversationState, AgentState, AgentStatus, SignPrediction, ConversationTurn


def test_conversation_state_initialization():
    state = ConversationState(session_id="test_session_1")
    assert state.session_id == "test_session_1"
    assert state.recognized_signs == []
    assert state.disambiguated_signs == []
    assert state.english_sentence is None
    assert state.agent_statuses == {}


def test_agent_status_updates():
    state = ConversationState(session_id="test_session_2")
    state.update_agent_status("VisionAgent", AgentState.RUNNING, 0.0)
    assert state.agent_statuses["VisionAgent"].state == AgentState.RUNNING

    state.update_agent_status("VisionAgent", AgentState.SUCCESS, 4.5, "Normalized 30 frames")
    assert state.agent_statuses["VisionAgent"].state == AgentState.SUCCESS
    assert state.agent_statuses["VisionAgent"].latency_ms == 4.5
    assert state.agent_statuses["VisionAgent"].message == "Normalized 30 frames"


def test_sign_prediction():
    pred = SignPrediction(sign="WANT", confidence=0.92, start_frame=0, end_frame=30)
    assert pred.sign == "WANT"
    assert pred.confidence == 0.92
    assert pred.start_frame == 0
    assert pred.end_frame == 30
