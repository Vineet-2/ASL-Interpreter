"""
Ambiguity resolution test suite evaluating ContextAgent against standard homonym test cases.
"""
import pytest
from src.agents.context_agent import ContextAgent
from src.models.schema import ConversationTurn
from src.api.server import AMBIGUITY_BENCHMARK_CASES


def test_ambiguity_benchmark_suite():
    agent = ContextAgent()
    passed = 0
    total = len(AMBIGUITY_BENCHMARK_CASES)

    for case in AMBIGUITY_BENCHMARK_CASES:
        raw = case["raw_signs"]
        expected = case["expected"]
        history_text = case.get("history_context")

        history = []
        if history_text:
            history = [ConversationTurn(
                turn_id=1,
                raw_signs=[],
                disambiguated_signs=[],
                english_translation=history_text
            )]

        resolved, notes = agent.disambiguate_sequence(raw, history)
        assert resolved == expected, f"Failed Case #{case['id']} ({case['description']}): got {resolved}, expected {expected}"
        passed += 1

    accuracy = (passed / total) * 100
    assert accuracy == 100.0
