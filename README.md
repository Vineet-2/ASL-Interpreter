# SignVoice — ASL Sign Recognition & Speech Assistant

A webcam-based system that recognizes a **fixed vocabulary of American Sign Language (ASL) signs** in real time and converts recognized sign sequences into spoken English.

**Scope, stated plainly:** this recognizes 59 signs (39 core + 20 extension classes, from the ASL Citizen dataset) — it is not an open-vocabulary or fully continuous ASL translator, and it does not model ASL grammar beyond the specific homonym-disambiguation rules listed below. Within that vocabulary, it continuously segments a live video stream into individual signs and uses an LLM to phrase the recognized sequence as a natural English sentence.

Built as six cooperating agents sharing a common "blackboard" state, each owning one stage of the pipeline: vision, sign recognition, context disambiguation, language generation, speech synthesis, and conversation memory. The multi-agent split exists so each stage can be swapped, benchmarked, or degraded independently — e.g. losing the LLM API doesn't take down sign recognition.

---

## System Architecture

```
graph TD
    subgraph Sensation & Recognition
        Webcam([Signer / Camera]) -->|Raw Video Frames| VisionAgent[1. Vision Agent: MediaPipe Hands / Holistic]
        VisionAgent -->|144D Normalized Keypoints & Geometric Anchors| SignAgent[2. Sign Recognition Agent: BiGRU + Attention]
    end

    subgraph Coordination & Reasoning
        SignAgent -->|Recognized Sign Sequence| Orchestrator[Orchestrator: Async Blackboard Coordinator]
        Orchestrator <-->|Session History / Turns| MemoryAgent[6. Conversation Memory Agent]
        Orchestrator -->|Raw Signs + History| ContextAgent[3. Context Agent: Rule Disambiguation]
        ContextAgent -->|Disambiguated Glosses| LanguageAgent[4. Language Agent: Groq / Gemini / Local]
        LanguageAgent -->|Fluent English Sentence| SpeechAgent[5. Speech Agent: pyttsx3 Offline TTS]
    end

    subgraph User Experience
        SpeechAgent -->|WAV Audio Base64 / Subtitles| WebUI([SignVoice Web UI / Testing Lab])
        Orchestrator -->|Live Status Telemetry & State| WebUI
    end
```

---

## Components

1. **Vision Agent** — extracts a 144-dimensional feature vector per frame (126 hand landmarks + 18 body-relative anchor features) using MediaPipe, with wrist-centric normalization and basic posture features.
2. **Sign Recognition Agent** — a 2-layer BiGRU with temporal self-attention pooling over a sliding window, classifying among the 59-sign vocabulary. Runs in PyTorch (`.pth`) or exported ONNX.
3. **Context Agent** — a rule-table disambiguator that uses recent conversation history to resolve a specific, known set of homonyms/polysemous signs (e.g. `EAT` vs `FOOD`, `SHOP` vs `STORE`, `ME` → `I`/`MY`). It only resolves cases it has an explicit rule for.
4. **Language Agent** — turns a disambiguated sign sequence into a fluent English sentence via a fallback chain: Groq (`llama-3.3-70b-versatile`) → Google Gemini → a local rule-based synthesizer if neither API key is configured or reachable.
5. **Speech Agent** — offline text-to-speech via `pyttsx3`, packaged as base64 audio for the browser, with a Web Speech API fallback.
6. **Memory Agent** — per-session conversation state and JSON transcript export.

---

## Measured Latency

Reported from the included benchmark suite (`tests/benchmark_latency.py`) — re-run and update these numbers for your own hardware before quoting them anywhere:

| Stage | SLA budget | Observed |
|---|---|---|
| Vision → sign classification | < 300 ms | ~7.9 ms  |
| Full pipeline (vision → spoken output) | < 2000 ms | ~103 ms |

---

## Known Limitations

- **Closed vocabulary.** Only the 59 signs in the ASL Citizen subset used for training are recognized — anything else is misclassified or ignored, not gracefully handled.
- **Rule-based disambiguation only.** The context agent resolves the homonym pairs explicitly coded into it; it does not generalize to new ambiguous signs.
- **LLM-dependent fluency.** Sentence quality is noticeably lower on the local rule-based fallback than when Groq/Gemini are reachable.
- **Untested generalization.** Not evaluated on signers, lighting, or camera setups outside the training/testing data.

---

## Quickstart

### 1. Clone & Install Dependencies

```
git clone <your-repo-url>
cd signvoice

python -m venv venv
venv\Scripts\activate  # Windows (or: source venv/bin/activate on Linux/macOS)

python -m pip install -r requirements.txt
```

### 2. Configure Environment Variables (optional)

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ASL_CITIZEN_VIDEOS=path/to/ASL_Citizen/videos
```

> If no external API keys are provided, SignVoice falls back to the local rule-based synthesizer.

### 3. Launch the Application

```
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

---

## Dataset & Training Pipeline

Trained on the **ASL Citizen** dataset (59 vocabulary classes: 39 core + 20 extension).

```
# 1. Extract landmarks from videos
python -m src.training.extract_landmarks --video-dir "path/to/ASL_Citizen/videos"

# 2. Train the BiGRU + Attention classifier
python -m src.training.train_bigru --epochs 30 --batch-size 32 --lr 0.001

# 3. Export to ONNX
python -m src.training.export_onnx
```

---

## Running Tests & Benchmarks

```
python -m pytest tests/ -v -o asyncio_mode=auto
python tests/benchmark_latency.py
```

Ambiguity-resolution benchmark (query the live endpoint):
```
python -c "import urllib.request, json; print(json.dumps(json.loads(urllib.request.urlopen('http://localhost:8000/api/benchmark/ambiguity').read()), indent=2))"
```

---

## REST & WebSocket API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web dashboard and testing lab |
| `/api/agents/status` | `GET` | Health, state, and latency metrics for all 6 agents |
| `/api/translate` | `POST` | Translates a raw sign gloss sequence into English text + speech audio |
| `/api/pipeline/run` | `POST` | Runs the full pipeline on batch landmarks or sign tokens |
| `/api/transcript/{session_id}` | `GET` | Retrieves JSON conversation history for a session |
| `/api/session/reset` | `POST` | Clears conversation state/memory for a session |
| `/api/benchmark/ambiguity` | `GET` | Evaluates accuracy against the coded homonym/ambiguity test cases |
| `/ws/stream` | `WebSocket` | Bi-directional streaming of video landmarks, subtitles, telemetry |

---

## Docker Deployment

```
docker-compose up --build
```

Access at `http://localhost:8000`.

---

## Project Structure

```
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── saved_models/              # Trained PyTorch (.pth) and ONNX models (git ignored)
├── src/
│   ├── config.py
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── vision_agent.py
│   │   ├── sign_agent.py
│   │   ├── context_agent.py
│   │   ├── language_agent.py
│   │   ├── speech_agent.py
│   │   └── memory_agent.py
│   ├── models/
│   │   ├── schema.py
│   │   └── bigru_model.py
│   ├── orchestration/
│   │   └── coordinator.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── extract_landmarks.py
│   │   ├── train_bigru.py
│   │   └── export_onnx.py
│   ├── data/
│   │   └── vocabulary.py
│   └── api/
│       └── server.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── tests/
    ├── test_schema.py
    ├── test_agents.py
    ├── test_ambiguity.py
    ├── test_geometric.py
    ├── test_live_server.py
    ├── test_pipeline.py
    └── benchmark_latency.py
```

---

## Dataset Attribution & Non-Commercial Notice

> [!IMPORTANT]
> **Personal & Non-Commercial Project Notice:**
> This repository and application is an independent **personal project developed strictly for non-commercial, educational, and research purposes**. It is not intended for commercial use or deployment.

### ASL Citizen Dataset Credit
This project utilizes and builds upon the **[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/)** dataset developed by **Microsoft Research** and academic collaborators in partnership with Deaf community members. We gratefully acknowledge and credit the creators, researchers, and community signers whose contributions made this dataset possible.

### Dataset Access & Contact
- This repository **does not host, redistribute, or license** the original ASL Citizen dataset or raw video files.
- If you wish to use, download, or access the ASL Citizen dataset for research or educational purposes, please visit the official [Microsoft Research ASL Citizen Project Page](https://www.microsoft.com/en-us/research/project/asl-citizen/) to review their data use agreement, submit an access request, or contact the authors directly.

### Citation
If you use the ASL Citizen dataset in your work, please cite their NeurIPS 2023 paper:

**Text Citation:**
> Aashaka Desai, Lauren Berger, Fyodor O. Minakov, Vanessa Milan, Chinmay Singh, Kriston L. Pumphrey, Richard Ladner, Hal Daumé III, Alex Xijie Lu, Naomi Caselli, and Danielle Bragg. *"ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition."* In *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023) Datasets and Benchmarks Track*, 2023.

**BibTeX:**
```bibtex
@inproceedings{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston L and Ladner, Richard and Daum{\'e} III, Hal and Lu, Alex Xijie and Caselli, Naomi and Bragg, Danielle},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=V2m925rV42}
}
```
