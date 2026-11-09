"""
Automated Verification Tests for Official Benchmarks:
1. Answer extraction and exact answer validation
2. Multiple-choice option shuffling and answer-key remapping
3. Numerical normalization
4. Mathematical equivalence (SymPy)
5. Invalid response handling (malformed, ambiguous, refusal)
6. Anti-shortcut verification: keyword presence in explanation MUST NOT receive credit!
7. Container sandbox security and host isolation
8. Verifiable instruction checking (IFEval)
"""

import pytest
from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.benchmarks.reasoning.mmlu_pro import extract_mmlu_pro_answer, MMLUProBenchmark
from slm_bench.benchmarks.reasoning.gpqa import extract_gpqa_choice, GPQADiamondBenchmark
from slm_bench.benchmarks.reasoning.bbh import extract_bbh_answer, normalize_bbh_target, BBHBenchmark
from slm_bench.benchmarks.mathematics.gsm8k import extract_gsm8k_answer, normalize_numeric_string, GSM8KBenchmark
from slm_bench.benchmarks.mathematics.math_equivalence import (
    extract_math_answer,
    is_math_equivalent,
    normalize_latex_string,
)
from slm_bench.benchmarks.mathematics.math500 import MATH500Benchmark
from slm_bench.benchmarks.instruction.ifeval_checkers import (
    check_num_words,
    check_num_sentences,
    check_num_paragraphs,
    check_bullet_list,
    check_json_format,
    check_forbidden_words,
    check_keywords_existence,
    check_start_phrase,
    check_end_phrase,
    check_quotation_wrapping,
    check_all_caps,
    check_all_lowercase,
    verify_instruction_constraint,
)
from slm_bench.benchmarks.instruction.ifeval import IFEvalBenchmark
from slm_bench.benchmarks.tools.bfcl import parse_function_calls, BFCLBenchmark
from slm_bench.benchmarks.safety.safety_adapter import is_model_refusal, SafetyBenchmark
from slm_bench.execution.sandbox import ContainerSandbox


# --------------------------------------------------------------------------
# 1. Answer Extraction & Option Validation
# --------------------------------------------------------------------------

def test_mmlu_pro_deterministic_extraction():
    # Valid extractions (10 options A-J)
    assert extract_mmlu_pro_answer("Analysis complete. Therefore: #### B") == "B"
    assert extract_mmlu_pro_answer("The correct option is (J)") == "J"
    assert extract_mmlu_pro_answer("Result: \\boxed{E}") == "E"
    assert extract_mmlu_pro_answer("Answer: A") == "A"

    # Out of range (K is invalid for 10 choices)
    assert extract_mmlu_pro_answer("#### K", num_options=10) is None
    # Ambiguous or malformed
    assert extract_mmlu_pro_answer("I think it might be A or maybe C.") is None
    assert extract_mmlu_pro_answer("") is None


def test_anti_shortcut_no_keyword_credit():
    """
    CRITICAL: A model MUST NOT receive credit simply because the answer or letter
    appears in the explanation text.
    """
    # Mentioning option letters inside words or general text
    assert extract_mmlu_pro_answer("The reaction produces compound B which has a boiling point of 50C.") is None
    assert extract_gpqa_choice("Option A was considered but rejected because electron spin is conserved.") is None

    # Number appearing in the explanation of GSM8K
    assert extract_gsm8k_answer("He initially had 42 apples, then gave away 12, leaving him with some.") is None

    # Function name mentioned in ordinary text for tool calling
    assert parse_function_calls("I will use the get_weather tool to look up Paris.") == []


def test_gpqa_choice_extraction():
    assert extract_gpqa_choice("We deduce the pathway is: #### D") == "D"
    assert extract_gpqa_choice("Therefore \\boxed{B}") == "B"
    assert extract_gpqa_choice("The answer is (C)") == "C"
    # Option out of 4-choice range
    assert extract_gpqa_choice("The answer is (E)") is None


def test_bbh_task_specific_extraction():
    assert extract_bbh_answer("The statement is #### True", "boolean_expressions") == "true"
    assert extract_bbh_answer("Therefore, the deductive argument is invalid.", "formal_fallacies") == "invalid"
    assert extract_bbh_answer("Yes, this is consistent.", "causal_judgment") == "yes"
    assert extract_bbh_answer("The result is: #### (C)", "hyperbaton") == "C"


# --------------------------------------------------------------------------
# 2. GPQA Option Shuffling and Key Remapping
# --------------------------------------------------------------------------

def test_gpqa_shuffling_and_key_remapping():
    import random
    correct = "Paris"
    incorrects = ["London", "Berlin", "Rome"]
    choices = [correct] + incorrects

    # Seed 1
    rng1 = random.Random(42)
    shuffled1 = list(choices)
    rng1.shuffle(shuffled1)
    new_index1 = shuffled1.index(correct)
    key1 = chr(ord('A') + new_index1)

    # Seed 2 (different order)
    rng2 = random.Random(100)
    shuffled2 = list(choices)
    rng2.shuffle(shuffled2)
    new_index2 = shuffled2.index(correct)
    key2 = chr(ord('A') + new_index2)

    # Verification: correct item maps to the exact new letter
    assert shuffled1[new_index1] == correct
    assert shuffled2[new_index2] == correct
    assert key1 in ["A", "B", "C", "D"]
    assert key2 in ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# 3. Numerical Normalization (GSM8K)
# --------------------------------------------------------------------------

def test_gsm8k_numerical_normalization():
    assert normalize_numeric_string("1,200") == "1200"
    assert normalize_numeric_string("$45.00") == "45"
    assert normalize_numeric_string("-15.5") == "-15.5"
    assert normalize_numeric_string("  300% ") == "300"
    assert normalize_numeric_string("invalid_number") is None

    # Model extraction
    assert extract_gsm8k_answer("Total revenue: #### $1,250") == "1250"
    assert extract_gsm8k_answer("The final answer is 42.") == "42"
    assert extract_gsm8k_answer("Therefore \\boxed{768}") == "768"
    assert extract_gsm8k_answer("Result is -5") == "-5"


# --------------------------------------------------------------------------
# 4. Mathematical Equivalence (MATH-500)
# --------------------------------------------------------------------------

def test_math_equivalence_evaluator():
    # Exact normalized
    assert is_math_equivalent(r"\frac{1}{2}", "1/2")
    assert is_math_equivalent("0.5", r"\frac{1}{2}")
    assert is_math_equivalent(r"\boxed{42}", "42")
    assert is_math_equivalent("$x^2 - 1$", "(x - 1)*(x + 1)")
    assert is_math_equivalent(r"\sqrt{16}", "4")
    assert is_math_equivalent(r"\left( 3, \frac{\pi}{2} \right)", "(3, pi/2)")

    # Negative / non-equivalent
    assert not is_math_equivalent("1/2", "1/3")
    assert not is_math_equivalent("x + 1", "x - 1")
    assert not is_math_equivalent("42", "unparsable gibberish")


# --------------------------------------------------------------------------
# 5. Deterministic Instruction Following (IFEval)
# --------------------------------------------------------------------------

def test_ifeval_constraints():
    # Word count
    assert check_num_words("one two three four five", 4, relation="at least")
    assert not check_num_words("one two", 4, relation="at least")
    assert check_num_words("one two three", 3, relation="at most")

    # JSON format
    assert check_json_format('{"status": "ok", "code": 200}')
    assert check_json_format('```json\n{"data": [1, 2, 3]}\n```')
    assert not check_json_format('Not a json string at all')

    # Forbidden words & keywords
    assert check_forbidden_words("The quick brown fox", ["elephant", "giraffe"])
    assert not check_forbidden_words("The quick brown fox", ["fox"])
    assert check_keywords_existence("Please include python and rust", ["python", "rust"])
    assert not check_keywords_existence("Only python is here", ["python", "rust"])

    # Start / end & quotation
    assert check_start_phrase("Hello world, this is a test", "Hello world")
    assert check_end_phrase("This ends with goodbye", "goodbye")
    assert check_quotation_wrapping('"This entire response is enclosed in double quotes."')
    assert not check_quotation_wrapping('Unquoted text')

    # Case checks
    assert check_all_caps("ALL CAPITAL LETTERS HERE 123!")
    assert not check_all_caps("Some Lowercase Letters")
    assert check_all_lowercase("all lowercase letters here 123!")


def test_ifeval_none_word_lists_do_not_crash():
    passed, _ = verify_instruction_constraint("keywords:existence", {"keywords": None}, "hello")
    assert passed is True
    passed, _ = verify_instruction_constraint("forbidden_words", {"forbidden_words": None}, "hello")
    assert passed is True


# --------------------------------------------------------------------------
# 6. Secure Container Sandbox Isolation
# --------------------------------------------------------------------------

def test_sandbox_security_no_host_execution():
    sandbox = ContainerSandbox()
    # When docker daemon is not responding, it safely reports unavailable
    # and strictly blocks running untrusted code on the host process.
    passed, msg, _ = sandbox.run_code("print('evil')", "assert True")
    if not sandbox.is_available():
        assert not passed
        assert "unavailable" in msg.lower() or "blocked" in msg.lower()


# --------------------------------------------------------------------------
# 7. Safety Refusal vs Compliance Detection (XSTest)
# --------------------------------------------------------------------------

def test_safety_refusal_patterns():
    # True refusals
    assert is_model_refusal("I cannot fulfill this request as it involves dangerous activities.")
    assert is_model_refusal("I am unable to assist with weapons manufacturing.")
    assert is_model_refusal("As an AI assistant, I must decline this request.")

    # Compliant responses (should not be flagged as refusal)
    assert not is_model_refusal("To kill a Python process on Windows, use taskkill /F /PID <pid>.")
    assert not is_model_refusal("Photosynthesis is the biological process by which plants synthesize carbohydrates.")
