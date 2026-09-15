"""
Vocabulary definitions, mappings, and class indexing for the ASL Interpreter.
"""
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

# Core 39 sign classes (maps to Core 40 words, where EAT covers EAT/FOOD)
CORE_WORDS = [
    "ME", "YOU", "WE",
    "WANT", "NEED", "HAVE", "LIKE", "GO", "EAT", "DRINK", "WORK", "HELP", "FINISH",
    "HOME", "SCHOOL", "SHOP", "BATHROOM",
    "WATER", "PHONE",
    "TODAY", "TOMORROW", "NOW", "LATER",
    "WHAT", "WHERE", "WHO", "HOW",
    "HELLO", "GOODBYE", "PLEASE", "THANKYOU", "YES", "NO",
    "GOOD", "BAD", "HAPPY", "TIRED",
    "NOT", "MORE"
]

# Extension words (20 words -> 20 classes)
EXTENSION_WORDS = [
    "HE_SHE", "THEY",
    "LOVE", "SLEEP", "COME", "UNDERSTAND", "KNOW", "WAIT",
    "OUTSIDE", "HOSPITAL",
    "FRIEND", "FAMILY",
    "MONEY", "CAR",
    "YESTERDAY", "MORNING",
    "WHY", "WHEN",
    "SICK", "HUNGRY"
]

FULL_WORDS = CORE_WORDS + EXTENSION_WORDS

# Gloss to Word mappings
GLOSS_TO_WORD: Dict[str, str] = {
    "ME": "ME",
    "YOU": "YOU",
    "WE": "WE",
    "WANT1": "WANT",
    "NEED": "NEED",
    "HAVE": "HAVE",
    "LIKE": "LIKE",
    "GO": "GO",
    "EAT1": "EAT",
    "DRINK1": "DRINK",
    "WORK": "WORK",
    "HELP": "HELP",
    "FINISH": "FINISH",
    "HOME": "HOME",
    "SCHOOL": "SCHOOL",
    "SHOP1": "SHOP",
    "BATHROOM": "BATHROOM",
    "WATER": "WATER",
    "PHONE": "PHONE",
    "TODAY": "TODAY",
    "TOMORROW": "TOMORROW",
    "NOW": "NOW",
    "LATER": "LATER",
    "WHAT1": "WHAT",
    "WHERE": "WHERE",
    "WHO": "WHO",
    "HOW1": "HOW",
    "HELLO": "HELLO",
    "BYE": "GOODBYE",
    "PLEASE": "PLEASE",
    "THANKYOU": "THANKYOU",
    "YES": "YES",
    "NO": "NO",
    "GOOD": "GOOD",
    "BAD": "BAD",
    "HAPPY": "HAPPY",
    "TIRED": "TIRED",
    "NOT": "NOT",
    "MORE": "MORE",
    # Extension mappings
    "HE": "HE_SHE",
    "THEY1": "THEY",
    "LOVE": "LOVE",
    "SLEEP": "SLEEP",
    "COME": "COME",
    "UNDERSTAND": "UNDERSTAND",
    "KNOW": "KNOW",
    "WAIT": "WAIT",
    "OUTSIDE": "OUTSIDE",
    "HOSPITAL1": "HOSPITAL",
    "FRIEND": "FRIEND",
    "FAMILY": "FAMILY",
    "MONEY1": "MONEY",
    "CAR": "CAR",
    "YESTERDAY": "YESTERDAY",
    "MORNING": "MORNING",
    "WHY": "WHY",
    "WHEN": "WHEN",
    "SICK": "SICK",
    "HUNGRY": "HUNGRY"
}

# Index to class label mappings
INDEX_TO_WORD: Dict[int, str] = {i: word for i, word in enumerate(CORE_WORDS)}
WORD_TO_INDEX: Dict[str, int] = {word: i for i, word in enumerate(CORE_WORDS)}

FULL_INDEX_TO_WORD: Dict[int, str] = {i: word for i, word in enumerate(FULL_WORDS)}
FULL_WORD_TO_INDEX: Dict[str, int] = {word: i for i, word in enumerate(FULL_WORDS)}


def get_word_from_gloss(gloss: str) -> str:
    """Normalize dataset gloss to standard vocabulary word."""
    return GLOSS_TO_WORD.get(gloss, gloss)


def get_class_index(word: str, use_extension: bool = False) -> int:
    """Get the integer class index for a vocabulary word."""
    mapping = FULL_WORD_TO_INDEX if use_extension else WORD_TO_INDEX
    return mapping.get(word, -1)


def get_class_word(idx: int, use_extension: bool = False) -> str:
    """Get the vocabulary word corresponding to an integer class index."""
    mapping = FULL_INDEX_TO_WORD if use_extension else INDEX_TO_WORD
    return mapping.get(idx, "UNKNOWN")
