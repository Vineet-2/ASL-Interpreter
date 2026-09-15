"""
Language Agent: Converts disambiguated ASL gloss sequences into fluent English sentences.
Uses a multi-tier fallback architecture: Groq (Primary) -> Gemini (Secondary) -> Local Heuristics (Offline).
"""
import os
import json
import logging
import httpx
from typing import List, Optional, Dict
from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, AgentState
from src.config import GROQ_API_KEY, GEMINI_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert ASL (American Sign Language) to English translator.
ASL has its own grammar (Topic-Comment, Time-First, omission of 'is/am/are' and articles).
Your task is to take a sequence of disambiguated ASL sign glosses and translate them into a natural, grammatically correct English sentence.

Rules:
1. Preserve the original meaning faithfully without hallucinating new information.
2. Insert appropriate articles ('a', 'the') and helping/copula verbs ('is', 'are', 'will', 'do') where appropriate.
3. Handle temporal markers properly (e.g. 'TOMORROW' -> future tense 'will', 'YESTERDAY' -> past tense, 'NOW' -> present continuous/simple).
4. Output STRICT JSON in the format: {"sentence": "<fluent English sentence>"} with no markdown or extra commentary.
"""


class LanguageAgent(BaseAgent):
    """
    Language Agent: Grammar translation with multi-provider resilience (Groq -> Gemini -> Local Rules).
    """
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        timeout: float = 2.0
    ):
        super().__init__(name="LanguageAgent", timeout=timeout)
        self.groq_key = groq_api_key or GROQ_API_KEY
        self.gemini_key = gemini_api_key or GEMINI_API_KEY

    async def translate_with_groq(self, signs: List[str]) -> Optional[str]:
        """Query Groq LLM API with tight latency timeout."""
        if not self.groq_key:
            return None

        prompt = f"ASL Signs: {' '.join(signs)}"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound-mini",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for model_name in models_to_try:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150,
                    "response_format": {"type": "json_object"}
                }
                try:
                    resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        return parsed.get("sentence")
                    else:
                        logger.debug(f"Groq model {model_name} returned status {resp.status_code}: {resp.text}")
                        continue
                except Exception as e:
                    logger.debug(f"Groq query exception with {model_name}: {e}")
                    continue
        return None

    async def translate_with_gemini(self, signs: List[str]) -> Optional[str]:
        """Query Google Gemini API as secondary fallback."""
        if not self.gemini_key:
            return None

        prompt = f"{SYSTEM_PROMPT}\n\nASL Signs: {' '.join(signs)}"
        gemini_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for gm in gemini_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent?key={self.gemini_key}"
                try:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        parsed = json.loads(raw_text)
                        return parsed.get("sentence")
                    elif resp.status_code == 404:
                        continue
                    else:
                        logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                        break
                except Exception as e:
                    logger.debug(f"Gemini query exception with {gm}: {e}")
                    break
        return None

    def translate_with_local_rules(self, signs: List[str]) -> str:
        """
        Deterministic, robust local grammatical synthesizer when offline or without API keys.
        """
        if not signs:
            return ""

        words = [s.upper() for s in signs]
        
        # Check for single word expressions
        if len(words) == 1:
            w = words[0]
            mappings = {
                "HELLO": "Hello!",
                "GOODBYE": "Goodbye!",
                "PLEASE": "Please.",
                "THANKYOU": "Thank you.",
                "YES": "Yes.",
                "NO": "No.",
                "HELP": "Help me, please.",
                "TIRED": "I am tired.",
                "HAPPY": "I am happy.",
                "GOOD": "Good.",
                "BAD": "Bad."
            }
            if w in mappings:
                return mappings[w]

        # Detect question markers
        is_question = any(q in words for q in ["WHAT", "WHERE", "WHO", "HOW", "WHY", "WHEN"])
        
        # Detect time markers
        time_marker = None
        for tm in ["TODAY", "TOMORROW", "NOW", "LATER", "YESTERDAY", "MORNING"]:
            if tm in words:
                time_marker = tm
                break

        # Common phrase combinations
        joined = " ".join(words)
        
        # Template rules
        templates: Dict[str, str] = {
            "I WANT GO STORE TODAY": "I want to go to the store today.",
            "I STORE TOMORROW GO": "I will go to the store tomorrow.",
            "I GO STORE TOMORROW": "I will go to the store tomorrow.",
            "BATHROOM WHERE": "Where is the bathroom?",
            "WHERE BATHROOM": "Where is the bathroom?",
            "THANKYOU I NEED HELP": "Thank you, I need help.",
            "I NOT WANT FOOD NOW": "I do not want food right now.",
            "WHAT YOU WANT": "What do you want?",
            "I FINISH WORK I WANT GO HOME": "I finished work and I want to go home.",
            "I TIRED I WANT SLEEP LATER": "I am tired and I want to sleep later.",
            "MY FRIEND HUNGRY WE WANT EAT": "My friend is hungry, we want to eat.",
            "YOU HAVE PHONE": "Do you have a phone?",
            "I NEED WATER": "I need water.",
            "WE WANT FOOD": "We want food.",
            "I LIKE SCHOOL": "I like school.",
            "YOU GO HOME NOW": "Are you going home now?"
        }
        
        if joined in templates:
            return templates[joined]

        # Heuristic assembly
        sentence_tokens = []
        has_negation = "NOT" in words
        
        # Subject extraction
        subject = "I"
        if "YOU" in words or "YOUR" in words:
            subject = "you"
        elif "WE" in words or "OUR" in words:
            subject = "we"
        elif "THEY" in words:
            subject = "they"
        elif "HE" in words:
            subject = "he"
        elif "SHE" in words:
            subject = "she"
        elif "MY" in words and "FRIEND" in words:
            subject = "my friend"

        # Tense helper
        tense_aux = ""
        if time_marker == "TOMORROW" or time_marker == "LATER":
            tense_aux = "will "
        elif time_marker == "YESTERDAY":
            tense_aux = "did " if has_negation else ""

        # Verb / predicate assembly
        filtered_words = [w for w in words if w not in ["ME", "I", "YOU", "WE", "THEY", "HE", "SHE", "NOT", time_marker] if w]
        predicate = " ".join(filtered_words).lower()
        
        # Normalize common predicate sequences
        predicate = predicate.replace("want go", "want to go to the")
        predicate = predicate.replace("go store", "go to the store")
        predicate = predicate.replace("go home", "go home")
        predicate = predicate.replace("go school", "go to school")
        predicate = predicate.replace("need help", "need help")
        predicate = predicate.replace("want food", "want food")
        predicate = predicate.replace("want eat", "want to eat")
        predicate = predicate.replace("need water", "need water")

        if is_question:
            q_word = next((q.capitalize() for q in ["Where", "What", "Who", "How", "Why", "When"] if q.upper() in words), "What")
            predicate_clean = " ".join([w for w in filtered_words if w not in ["WHAT", "WHERE", "WHO", "HOW", "WHY", "WHEN"]]).lower()
            if "bathroom" in predicate_clean:
                return "Where is the bathroom?"
            return f"{q_word} do {subject} {predicate_clean}?".strip()

        negation_str = "do not " if (has_negation and not tense_aux) else ("not " if has_negation else "")
        time_suffix = f" {time_marker.lower()}" if time_marker else ""
        
        capital_subject = subject.capitalize()
        assembled = f"{capital_subject} {tense_aux}{negation_str}{predicate}{time_suffix}.".strip()
        # Clean double spaces
        assembled = " ".join(assembled.split())
        return assembled

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Translates state.disambiguated_signs into state.english_sentence.
        """
        signs = state.disambiguated_signs
        if not signs:
            # Check recognized signs
            signs = [p.sign for p in state.recognized_signs]

        if not signs:
            state.english_sentence = None
            state.translation_provider = None
            return state

        # 1. Try Groq
        try:
            translation = await self.translate_with_groq(signs)
            if translation:
                state.english_sentence = translation
                state.translation_provider = "groq"
                return state
        except Exception as e:
            logger.warning(f"Groq translation failed: {e}")

        # 2. Try Gemini fallback
        try:
            translation = await self.translate_with_gemini(signs)
            if translation:
                state.english_sentence = translation
                state.translation_provider = "gemini"
                return state
        except Exception as e:
            logger.warning(f"Gemini translation fallback failed: {e}")

        # 3. Local rules fallback
        local_translation = self.translate_with_local_rules(signs)
        state.english_sentence = local_translation
        state.translation_provider = "local_rules"
        return state

    async def fallback(self, state: ConversationState, error_msg: str) -> ConversationState:
        """Fallback: Use raw disambiguated signs as fallback text instead of failing."""
        raw_text = " ".join(state.disambiguated_signs or [p.sign for p in state.recognized_signs])
        state.english_sentence = raw_text
        state.translation_provider = "raw_fallback"
        return state
