"""
Unit tests for each of the 6 individual agents.
"""
import pytest
import numpy as np
from src.models.schema import ConversationState, SignPrediction, ConversationTurn
from src.agents.vision_agent import VisionAgent
from src.agents.sign_agent import SignRecognitionAgent
from src.agents.context_agent import ContextAgent
from src.agents.language_agent import LanguageAgent
from src.agents.speech_agent import SpeechAgent
from src.agents.memory_agent import ConversationMemoryAgent


@pytest.mark.asyncio
async def test_vision_agent():
    agent = VisionAgent()
    state = ConversationState()
    # Mock single frame feature vector (126 dims)
    mock_frame = np.random.randn(126).tolist()
    state.raw_landmarks = [mock_frame]

    result = await agent.run(state)
    assert result.agent_statuses["VisionAgent"].state.value == "SUCCESS"
    assert result.landmark_shape is not None


@pytest.mark.asyncio
async def test_sign_recognition_agent():
    agent = SignRecognitionAgent()
    state = ConversationState()
    # Mock 30-frame sequence (30, 126)
    mock_seq = np.random.randn(30, 126).tolist()
    state.raw_landmarks = mock_seq

    result = await agent.run(state)
    assert result.agent_statuses["SignRecognitionAgent"].state.value == "SUCCESS"


@pytest.mark.asyncio
async def test_context_agent_disambiguation():
    agent = ContextAgent()
    state = ConversationState()
    state.disambiguated_signs = ["ME", "WANT", "EAT", "NOW"]

    result = await agent.run(state)
    assert result.disambiguated_signs == ["I", "WANT", "FOOD", "NOW"]
    assert len(result.context_notes) > 0


@pytest.mark.asyncio
async def test_language_agent_local_translation():
    agent = LanguageAgent()
    state = ConversationState()
    state.disambiguated_signs = ["I", "WANT", "GO", "STORE", "TODAY"]

    result = await agent.run(state)
    assert result.english_sentence is not None
    assert "store" in result.english_sentence.lower()
    assert result.translation_provider in ["groq", "gemini", "local_rules"]


@pytest.mark.asyncio
async def test_speech_agent():
    agent = SpeechAgent()
    state = ConversationState()
    state.english_sentence = "Hello, world!"

    result = await agent.run(state)
    assert result.agent_statuses["SpeechAgent"].state.value in ["SUCCESS", "DEGRADED"]


@pytest.mark.asyncio
async def test_memory_agent():
    agent = ConversationMemoryAgent()
    state = ConversationState(session_id="session_test")
    state.disambiguated_signs = ["I", "NEED", "HELP"]
    state.english_sentence = "I need help."

    result = await agent.run(state)
    assert len(result.history) == 1
    assert result.history[0].english_translation == "I need help."


def test_landmark_sequence_dataset_augmentations():
    from src.training.dataset import LandmarkSequenceDataset
    
    # 2 sample sequences with dummy hand landmarks
    sample_seq1 = np.random.randn(25, 126).astype(np.float32)
    sample_seq2 = np.random.randn(40, 126).astype(np.float32)
    labels = [0, 1]

    dataset = LandmarkSequenceDataset(
        samples=[sample_seq1, sample_seq2],
        labels=labels,
        target_seq_len=30,
        augment=True
    )

    assert len(dataset) == 2
    for i in range(len(dataset)):
        tensor_x, label_y = dataset[i]
        assert tensor_x.shape == (30, 126)
        assert isinstance(label_y, int)

