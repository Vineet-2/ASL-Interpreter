"""
Speech Agent: Text-to-Speech synthesis using pyttsx3 with base64 audio serialization for client delivery.
"""
import os
import base64
import tempfile
import logging
from typing import Optional
from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, AgentState

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

logger = logging.getLogger(__name__)


class SpeechAgent(BaseAgent):
    """
    Speech Agent: Synthesizes English sentences into spoken audio (WAV) encoded in Base64.
    """
    def __init__(self, voice_rate: int = 160, volume: float = 1.0, timeout: float = 1.0):
        super().__init__(name="SpeechAgent", timeout=timeout)
        self.voice_rate = voice_rate
        self.volume = volume

    def synthesize_wav_base64(self, text: str) -> Optional[str]:
        """
        Synthesize text to a temporary WAV file and return Base64 encoded string.
        """
        if not HAS_PYTTSX3 or not text.strip():
            return None

        temp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name

            engine = pyttsx3.init()
            engine.setProperty('rate', self.voice_rate)
            engine.setProperty('volume', self.volume)
            engine.save_to_file(text, temp_wav)
            engine.runAndWait()

            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                with open(temp_wav, "rb") as audio_file:
                    encoded = base64.b64encode(audio_file.read()).decode("utf-8")
                return encoded
        except Exception as e:
            logger.warning(f"pyttsx3 synthesis error: {e}")
            return None
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

        return None

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Synthesizes state.english_sentence into base64 audio payload.
        """
        if not state.english_sentence:
            state.speech_audio_base64 = None
            return state

        try:
            audio_b64 = self.synthesize_wav_base64(state.english_sentence)
            state.speech_audio_base64 = audio_b64
            state.tts_format = "wav"
        except Exception as e:
            logger.warning(f"Speech synthesis error in agent: {e}")
            state.speech_audio_base64 = None
            
        return state

    async def fallback(self, state: ConversationState, error_msg: str) -> ConversationState:
        """Graceful degradation: system continues with text only if TTS fails."""
        state.speech_audio_base64 = None
        return state
