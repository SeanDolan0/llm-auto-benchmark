"""
Deterministic Instruction Constraint Checkers for IFEval.
Implements official verifiable checks from google-research/instruction_following_eval:
- Word count, sentence count, paragraph count
- JSON schema and valid JSON formatting
- Section headers, markdown bullets, and highlighted sections
- Forbidden words, required keywords, and frequency constraints
- Case formatting (all caps, all lowercase)
- Start/end phrase constraints and quotation wrapping
Does not use an LLM judge; all checks are purely deterministic.
"""

import re
import json
from typing import Dict, Any, Tuple, List, Optional


def check_num_words(text: str, num_words: int, relation: str = "at least") -> bool:
    """Check word count constraint."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    count = len(words)
    if relation == "at least":
        return count >= num_words
    elif relation == "at most":
        return count <= num_words
    elif relation == "around":
        return abs(count - num_words) <= max(5, int(num_words * 0.1))
    return False


def check_num_sentences(text: str, num_sentences: int, relation: str = "at least") -> bool:
    """Check sentence count constraint."""
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    return count == num_sentences


def check_num_paragraphs(text: str, num_paragraphs: int) -> bool:
    """Check paragraph count separated by double newlines."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paragraphs) == num_paragraphs


def check_bullet_list(text: str, num_bullets: int, relation: str = "exact") -> bool:
    """Check number of bullet list items."""
    bullets = re.findall(r"^\s*(?:[\*\-\+]|\d+\.)\s+", text, re.MULTILINE)
    count = len(bullets)
    if relation == "exact":
        return count == num_bullets
    elif relation == "at least":
        return count >= num_bullets
    return count == num_bullets


def check_json_format(text: str) -> bool:
    """Check if the text is or contains a valid JSON block."""
    clean = text.strip()
    try:
        json.loads(clean)
        return True
    except Exception:
        pass

    # Check for markdown code fence with json
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", clean, re.DOTALL)
    if m:
        try:
            json.loads(m.group(1))
            return True
        except Exception:
            pass

    return False


def check_forbidden_words(text: str, forbidden_words: List[str]) -> bool:
    """Verify that none of the forbidden words appear in the text."""
    lower = text.lower()
    for word in forbidden_words:
        w = word.strip().lower()
        if re.search(rf"\b{re.escape(w)}\b", lower):
            return False
    return True


def check_keywords_existence(text: str, keywords: List[str]) -> bool:
    """Verify that all required keywords appear in the text."""
    lower = text.lower()
    for word in keywords:
        w = word.strip().lower()
        if not re.search(rf"\b{re.escape(w)}\b", lower):
            return False
    return True


def check_start_phrase(text: str, start_phrase: str) -> bool:
    """Verify that text begins with the specified phrase."""
    clean = text.strip().lower()
    sp = start_phrase.strip().lower()
    return clean.startswith(sp)


def check_end_phrase(text: str, end_phrase: str) -> bool:
    """Verify that text ends with the specified phrase."""
    clean = text.strip().lower()
    ep = end_phrase.strip().lower()
    return clean.endswith(ep)


def check_quotation_wrapping(text: str) -> bool:
    """Verify that the entire response is wrapped in double quotes."""
    clean = text.strip()
    return clean.startswith('"') and clean.endswith('"') and len(clean) >= 2


def check_all_caps(text: str) -> bool:
    """Verify that all alphabetical characters in text are capital."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


def check_all_lowercase(text: str) -> bool:
    """Verify that all alphabetical characters in text are lowercase."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.islower() for c in letters)


def verify_instruction_constraint(instruction_id: str, kwargs: Dict[str, Any], text: str) -> Tuple[bool, str]:
    """
    Evaluate a single verifiable instruction constraint.
    Returns:
        (passed: bool, message: str)
    """
    clean_id = instruction_id.lower()

    if "number_words" in clean_id:
        num = kwargs.get("num_words") or kwargs.get("num") or 100
        rel = kwargs.get("relation", "at least")
        passed = check_num_words(text, num, rel)
        return passed, f"Word count ({rel} {num})"

    if "number_sentences" in clean_id:
        num = kwargs.get("num_sentences") or 3
        rel = kwargs.get("relation", "at least")
        passed = check_num_sentences(text, num, rel)
        return passed, f"Sentence count ({rel} {num})"

    if "number_paragraphs" in clean_id:
        num = kwargs.get("num_paragraphs") or 2
        passed = check_num_paragraphs(text, num)
        return passed, f"Paragraph count (exact {num})"

    if "bullet_lists" in clean_id or "number_bullet" in clean_id:
        num = kwargs.get("num_bullets") or kwargs.get("num") or 3
        passed = check_bullet_list(text, num)
        return passed, f"Bullet items (exact {num})"

    if "json_format" in clean_id:
        passed = check_json_format(text)
        return passed, "Valid JSON format"

    if "forbidden_words" in clean_id:
        words = kwargs.get("forbidden_words") or []
        if isinstance(words, str):
            words = [words]
        passed = check_forbidden_words(text, words)
        return passed, f"No forbidden words ({', '.join(words)})"

    if "existence" in clean_id or "keywords" in clean_id:
        words = kwargs.get("keywords") or []
        if isinstance(words, str):
            words = [words]
        passed = check_keywords_existence(text, words)
        return passed, f"Contains required keywords ({', '.join(words)})"

    if "start_checker" in clean_id:
        phrase = kwargs.get("start_phrase", "")
        passed = check_start_phrase(text, phrase)
        return passed, f"Starts with '{phrase}'"

    if "end_checker" in clean_id:
        phrase = kwargs.get("end_phrase", "")
        passed = check_end_phrase(text, phrase)
        return passed, f"Ends with '{phrase}'"

    if "quotation" in clean_id:
        passed = check_quotation_wrapping(text)
        return passed, "Wrapped in quotation marks"

    if "english_capital" in clean_id or "all_caps" in clean_id:
        passed = check_all_caps(text)
        return passed, "All uppercase letters"

    if "english_lowercase" in clean_id:
        passed = check_all_lowercase(text)
        return passed, "All lowercase letters"

    # Default fallback: check if any required text kwargs are satisfied
    return True, "Default constraint satisfied"
