import asyncio
import time
import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestration.coordinator import Orchestrator
from src.models.schema import ConversationState


async def run_latency_benchmark(num_iterations: int = 15):
    orchestrator = Orchestrator()
    print(f"\n=======================================================")
    print(f"   ASL ASSISTANT: MULTI-AGENT LATENCY BENCHMARK        ")
    print(f"=======================================================")
    print(f"Target Budgets:")
    print(f"  - Vision -> Sign Path:  < 300 ms")
    print(f"  - Total Pipeline Path:  < 2000 ms\n")

    latencies = []
    agent_latencies = {
        "VisionAgent": [],
        "SignRecognitionAgent": [],
        "ContextAgent": [],
        "LanguageAgent": [],
        "SpeechAgent": [],
        "ConversationMemoryAgent": []
    }

    # Warmup
    warmup_state = ConversationState(session_id="warmup")
    warmup_state.disambiguated_signs = ["ME", "STORE", "TOMORROW", "GO"]
    await orchestrator.run_pipeline(warmup_state)

    for i in range(num_iterations):
        state = ConversationState(session_id=f"bench_{i}")
        # 30-frame simulated landmark window
        state.raw_landmarks = np.random.randn(30, 126).tolist()
        state.disambiguated_signs = ["ME", "WANT", "EAT", "NOW"]

        t0 = time.perf_counter()
        result = await orchestrator.run_pipeline(state)
        t_total = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_total)

        for name, status in result.agent_statuses.items():
            if name in agent_latencies:
                agent_latencies[name].append(status.latency_ms)

    avg_total = np.mean(latencies)
    p95_total = np.percentile(latencies, 95)
    vision_sign_path = np.mean(agent_latencies["VisionAgent"]) + np.mean(agent_latencies["SignRecognitionAgent"])

    print(f"--- Results over {num_iterations} iterations ---")
    for name, times in agent_latencies.items():
        print(f"  * {name:<26}: Avg = {np.mean(times):.2f} ms | P95 = {np.percentile(times, 95):.2f} ms")

    print(f"\nPath Summaries:")
    print(f"  * Vision -> Sign Path Avg:  {vision_sign_path:.2f} ms  (Budget: < 300 ms)  => {'[PASS]' if vision_sign_path < 300 else '[FAIL]'}")
    print(f"  * End-to-End Pipeline Avg:   {avg_total:.2f} ms  (Budget: < 2000 ms) => {'[PASS]' if avg_total < 2000 else '[FAIL]'}")
    print(f"  * P95 End-to-End Pipeline:   {p95_total:.2f} ms\n")
    print(f"=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_latency_benchmark())
