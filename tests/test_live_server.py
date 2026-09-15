"""
Live API verification test script for FastAPI server.
"""
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_endpoints():
    # 1. Agent status
    resp = client.get("/api/agents/status")
    assert resp.status_code == 200
    status = resp.json()
    assert "agents" in status
    print("1. /api/agents/status OK:", list(status["agents"].keys()))

    # 2. Ambiguity benchmark
    resp = client.get("/api/benchmark/ambiguity")
    assert resp.status_code == 200
    bench = resp.json()
    assert bench["context_agent_accuracy_pct"] == 100.0
    print(f"2. /api/benchmark/ambiguity OK: Context Acc {bench['context_agent_accuracy_pct']}%")

    # 3. Direct translation
    resp = client.post("/api/translate", json={"signs": ["ME", "STORE", "TOMORROW", "GO"]})
    assert resp.status_code == 200
    trans = resp.json()
    assert trans["english_sentence"] is not None
    print(f"3. /api/translate OK: \"{trans['english_sentence']}\"")

    # 4. Homonym disambiguation
    resp2 = client.post("/api/translate", json={"signs": ["ME", "WANT", "EAT", "NOW"]})
    assert resp2.status_code == 200
    trans2 = resp2.json()
    assert "FOOD" in trans2["disambiguated_signs"]
    print(f"4. /api/translate (Homonym EAT -> FOOD) OK: {trans2['disambiguated_signs']}")

    # 5. Transcript query
    resp3 = client.get("/api/transcript/default_session")
    assert resp3.status_code == 200
    transcript = resp3.json()
    print(f"5. /api/transcript/default_session OK: turns={transcript['turn_count']}")


if __name__ == "__main__":
    test_endpoints()

