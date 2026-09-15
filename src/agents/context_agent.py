"""
Context Agent: Disambiguates homonyms, polysemous signs, and grammatical roles
using conversation history and a structured rule table.
"""
import re
from typing import List, Dict, Tuple, Optional
import logging
from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, ConversationTurn

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    """
    Context Agent: Resolves ambiguity in raw recognized signs
    (e.g., EAT vs FOOD, SHOP vs STORE, I vs ME vs MY, WORK verb vs noun)
    based on immediate syntactic co-occurrence and multi-turn conversation history.
    """
    def __init__(self, timeout: float = 0.5):
        super().__init__(name="ContextAgent", timeout=timeout)
        self.rules: Dict[str, Dict] = self._init_rule_table()

    def _init_rule_table(self) -> Dict[str, Dict]:
        """
        Rule table defining disambiguation triggers, contexts, and target resolutions.
        """
        return {
            "EAT": {
                "default": "EAT",
                "rules": [
                    # If preceded by WANT/NEED/LIKE/HAVE without another noun, or preceded by MORE/GOOD/BAD
                    {"preceding": ["WANT", "NEED", "MORE", "HAVE", "LIKE", "BUY"], "following": [], "resolve": "FOOD", "note": "Nominalized food object"},
                    {"preceding": ["GOOD", "BAD", "DELICIOUS"], "following": [], "resolve": "FOOD", "note": "Adjective modifying food noun"},
                    {"preceding": ["STORE", "SHOP"], "following": [], "resolve": "FOOD", "note": "Food purchased at store"},
                    {"preceding": ["GO"], "following": [], "resolve": "EAT", "note": "Infinitive action: go to eat"}
                ]
            },
            "SHOP": {
                "default": "STORE",
                "rules": [
                    {"preceding": ["GO", "AT", "IN", "TO"], "following": [], "resolve": "STORE", "note": "Location destination: store"},
                    {"preceding": ["LIKE", "WANT", "TIME"], "following": ["TODAY", "TOMORROW", "NOW", "LATER"], "resolve": "SHOP", "note": "Action verb: to shop"}
                ]
            },
            "WORK": {
                "default": "WORK",
                "rules": [
                    {"preceding": ["GO", "AT", "FROM", "FINISH"], "following": [], "resolve": "WORKPLACE", "note": "Work as location/employment"},
                    {"preceding": ["ME", "YOU", "WE", "THEY", "HE_SHE"], "following": ["TODAY", "TOMORROW", "NOW", "HARD"], "resolve": "WORK", "note": "Action verb: to work"}
                ]
            },
            "DRINK": {
                "default": "DRINK",
                "rules": [
                    {"preceding": ["WANT", "NEED", "HAVE", "MORE"], "following": [], "resolve": "BEVERAGE", "note": "Drink as noun (beverage)"},
                    {"preceding": ["ME", "YOU", "WE", "GO"], "following": ["WATER"], "resolve": "DRINK", "note": "Drink as action"}
                ]
            },
            "ME": {
                "default": "I",
                "rules": [
                    {"preceding": ["HELP", "CALL", "TELL", "GIVE", "FOR", "WITH"], "following": [], "resolve": "ME", "note": "Objective case (object of verb/prep)"},
                    {"preceding": [], "following": ["PHONE", "HOME", "CAR", "FAMILY", "FRIEND", "MONEY"], "resolve": "MY", "note": "Possessive determiner before noun"},
                    {"preceding": [], "following": ["WANT", "NEED", "HAVE", "LIKE", "GO", "WORK", "EAT", "FINISH"], "resolve": "I", "note": "Nominative subject before verb"}
                ]
            },
            "YOU": {
                "default": "YOU",
                "rules": [
                    {"preceding": [], "following": ["PHONE", "HOME", "CAR", "FAMILY", "FRIEND", "MONEY"], "resolve": "YOUR", "note": "Possessive determiner before noun"}
                ]
            },
            "WE": {
                "default": "WE",
                "rules": [
                    {"preceding": [], "following": ["PHONE", "HOME", "CAR", "FAMILY", "FRIEND", "MONEY"], "resolve": "OUR", "note": "Possessive determiner"}
                ]
            },
            "HE_SHE": {
                "default": "THEY",
                "rules": [
                    {"history_keywords": ["MOTHER", "SISTER", "GIRL", "WOMAN", "SHE", "HER"], "resolve": "SHE", "note": "Anaphoric resolution to female pronoun"},
                    {"history_keywords": ["FATHER", "BROTHER", "BOY", "MAN", "HE", "HIM"], "resolve": "HE", "note": "Anaphoric resolution to male pronoun"}
                ]
            }
        }

    def disambiguate_sequence(
        self,
        signs: List[str],
        history: Optional[List[ConversationTurn]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Runs rule-based disambiguation on the sequence of recognized signs.
        """
        if not signs:
            return [], []

        history_text = ""
        if history:
            history_text = " ".join([turn.english_translation.upper() + " " + " ".join(turn.raw_signs) for turn in history]).upper()

        disambiguated = []
        notes = []

        for i, sign in enumerate(signs):
            upper_sign = sign.upper()
            if upper_sign not in self.rules:
                disambiguated.append(upper_sign)
                continue

            rule_entry = self.rules[upper_sign]
            resolved = rule_entry["default"]
            applied_note = f"{upper_sign} -> {resolved} (default)"

            # Check individual rules
            preceding_signs = [s.upper() for s in signs[:i]]
            following_signs = [s.upper() for s in signs[i+1:]]

            for r in rule_entry.get("rules", []):
                match = True
                
                # Check preceding requirements
                if "preceding" in r and r["preceding"]:
                    if not preceding_signs or preceding_signs[-1] not in r["preceding"]:
                        match = False
                        
                # Check following requirements
                if "following" in r and r["following"]:
                    if not following_signs or following_signs[0] not in r["following"]:
                        match = False
                        
                # Check multi-turn conversation history with word boundary
                if "history_keywords" in r and r["history_keywords"]:
                    if not any(re.search(rf"\b{re.escape(kw)}\b", history_text) for kw in r["history_keywords"]):
                        match = False
                        
                if match:
                    resolved = r["resolve"]
                    applied_note = f"{upper_sign} -> {resolved} ({r.get('note', 'rule matched')})"
                    break

            disambiguated.append(resolved)
            notes.append(applied_note)

        return disambiguated, notes

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Disambiguates state.recognized_signs or existing state.disambiguated_signs.
        """
        raw_signs = [p.sign for p in state.recognized_signs] if state.recognized_signs else state.disambiguated_signs
        
        if raw_signs:
            disambiguated, notes = self.disambiguate_sequence(raw_signs, state.history)
            state.disambiguated_signs = disambiguated
            state.context_notes = notes
        else:
            state.disambiguated_signs = []
            state.context_notes = []
            
        return state
