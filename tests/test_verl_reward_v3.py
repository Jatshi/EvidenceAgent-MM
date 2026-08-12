from scripts.verl_reward_v3 import compute_score

GROUND_TRUTH = {
    "status": "answered",
    "citation_ids": ["session:utt:1", "session:ocr:5"],
}


def test_exact_json_receives_full_reward() -> None:
    prediction = '{"status":"answered","citation_ids":["session:utt:1","session:ocr:5"]}'
    assert 0.98 < compute_score("evidenceagent_mm_v3", prediction, GROUND_TRUTH) <= 1.0


def test_terminal_json_is_extracted_from_agent_transcript() -> None:
    transcript = (
        '<tool_call>{"query":"design B","top_k":4}</tool_call>\n'
        "tool output: untrusted evidence\n"
        "```json\n"
        '{"status":"answered","citation_ids":'
        '["session:utt:1","session:ocr:5"]}\n```'
    )
    assert 0.98 < compute_score("evidenceagent_mm_v3", transcript, GROUND_TRUTH) <= 1.0


def test_malformed_early_rollout_gets_partial_not_correctness_reward() -> None:
    prediction = "status answered; citation_ids include session:utt:1"
    score = compute_score("evidenceagent_mm_v3", prediction, GROUND_TRUTH)
    assert 0.0 < score < 1.0


def test_unsafe_answer_is_penalized() -> None:
    ground_truth = {
        "status": "abstained",
        "citation_ids": [],
        "must_not_answer": True,
    }
    unsafe = '{"status":"answered","citation_ids":[]}'
    safe = '{"status":"abstained","citation_ids":[]}'
    assert compute_score("evidenceagent_mm_v3", safe, ground_truth) > compute_score(
        "evidenceagent_mm_v3", unsafe, ground_truth
    )


def test_efficiency_breaks_ties_without_overriding_correctness() -> None:
    concise = '{"status":"answered","citation_ids":["wrong"]}'
    verbose = "analysis " * 80 + concise
    exact_but_verbose = "analysis " * 80 + (
        '{"status":"answered","citation_ids":["session:utt:1","session:ocr:5"]}'
    )
    assert compute_score("evidenceagent_mm_v3", concise, GROUND_TRUTH) > compute_score(
        "evidenceagent_mm_v3", verbose, GROUND_TRUTH
    )
    assert compute_score("evidenceagent_mm_v3", exact_but_verbose, GROUND_TRUTH) > compute_score(
        "evidenceagent_mm_v3", concise, GROUND_TRUTH
    )
