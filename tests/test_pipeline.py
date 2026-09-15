"""
Integration test for full Orchestrator blackboard pipeline execution and failure isolation.
"""
import pytest
from src.orchestration.coordinator import Orchestrator
from src.models.schema import ConversationState, AgentState


@pytest.mark.asyncio
async def test_full_pipeline_success():
    orchestrator = Orchestrator()
    state = ConversationState(session_id="integration_test_session")
    state.disambiguated_signs = ["ME", "STORE", "TOMORROW", "GO"]

    result = await orchestrator.run_pipeline(state)
    assert result.english_sentence is not None
    assert result.total_pipeline_latency_ms > 0
    assert result.agent_statuses["ContextAgent"].state == AgentState.SUCCESS
    assert result.agent_statuses["LanguageAgent"].state in [AgentState.SUCCESS, AgentState.DEGRADED]
    assert result.agent_statuses["ConversationMemoryAgent"].state == AgentState.SUCCESS


@pytest.mark.asyncio
async def test_pipeline_failure_containment():
    orchestrator = Orchestrator()
    # Mock speech agent failing
    orchestrator.speech.synthesize_wav_base64 = lambda text: (_ for _ in ()).throw(RuntimeError("Simulated TTS audio error"))
    
    state = ConversationState(session_id="fault_tolerant_session")
    state.disambiguated_signs = ["HELLO"]

    # Pipeline should complete without crashing
    result = await orchestrator.run_pipeline(state)
    assert result.english_sentence in ["Hello!", "Hello.", "Hello"]
    assert result.agent_statuses["SpeechAgent"].state in [AgentState.ERROR, AgentState.DEGRADED, AgentState.SUCCESS]


@pytest.mark.asyncio
async def test_concurrent_sessions_vision_isolation():
    """Verify that multiple concurrent sessions maintain isolated vision buffers and smoothing states."""
    import numpy as np
    orchestrator = Orchestrator()

    from tests.test_geometric import _open_right_hand, _bunched_eat_hand
    frame_a = _open_right_hand().tolist()
    frame_b = _bunched_eat_hand().tolist()

    state_a = ConversationState(session_id="user_alice", raw_landmarks=[frame_a])
    state_b = ConversationState(session_id="user_bob", raw_landmarks=[frame_b])

    res_a = await orchestrator.run_pipeline(state_a)
    res_b = await orchestrator.run_pipeline(state_b)

    vision_a = orchestrator._get_vision("user_alice")
    vision_b = orchestrator._get_vision("user_bob")

    assert len(vision_a.buffer) == 1
    assert len(vision_b.buffer) == 1
    assert not np.allclose(vision_a.buffer[0], vision_b.buffer[0])


@pytest.mark.asyncio
async def test_session_eviction_and_cleanup():
    """Verify that stale sessions are evicted and bounded state limit is enforced."""
    orchestrator = Orchestrator(max_sessions=3, session_ttl_sec=0.05)

    # Access 3 distinct sessions
    orchestrator._get_vision("session_1")
    orchestrator._get_vision("session_2")
    orchestrator._get_vision("session_3")
    assert len(orchestrator.session_last_active) == 3

    # Access a 4th session -> triggers cleanup and LRU eviction
    orchestrator._get_vision("session_4")
    assert len(orchestrator.session_last_active) <= 3

    # Wait for TTL expiration
    import asyncio
    await asyncio.sleep(0.06)

    evicted_count = orchestrator.cleanup_stale_sessions(max_age_seconds=0.05)
    assert evicted_count >= 1
    assert len(orchestrator.vision_agents) == 0
