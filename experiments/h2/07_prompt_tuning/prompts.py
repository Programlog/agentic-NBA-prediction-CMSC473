"""Prompt variants for H2 experiment 07.

V0 is the unchanged 05 anchored single-analyst prompt. V1-V4 replace the CrewAI
agent's goal/task instructions (``agents.INSTRUCTIONS``/``EXPECTED_OUTPUT``) with a
rewritten version and add evidence to the agent role:

  V1  rewritten instructions + effect table (within-distance-band training rates)
  V2  V1 + similar training shots (kNN-50 rate and the 5 nearest, with outcomes)
  V3  V1 + the frozen XGBoost estimate, presented as a statistical-model tool output
  V4  V1 + similar shots + XGBoost
"""

from __future__ import annotations

import contextlib

from nba_shots import agents

VARIANTS = ["V0", "V1", "V2", "V3", "V4"]
EVIDENCE = {
    "V0": [],
    "V1": ["effects"],
    "V2": ["effects", "knn"],
    "V3": ["effects", "model"],
    "V4": ["effects", "knn", "model"],
}

INSTRUCTIONS_V1 = (
    "Estimate the probability that this NBA field-goal attempt is made, using only the "
    "supplied evidence (all measured before release; null means unknown). Write your "
    "reasoning in rationale BEFORE giving p_make, in this order. "
    "(1) Base rate: start from the training make rate for this shot distance. "
    "(2) Adjust: for each other factor, compare its within-band rate with the band rate and "
    "move only part of the way, because the factors overlap; do not add every difference "
    "in full. "
    "(3) Extra evidence, if supplied: the similar-shots rate is a direct but noisy estimate "
    "(about 50 shots), and the 5 listed examples are too few to count on their own. The "
    "statistical model estimate combines all nine measurements on 91k shots and is "
    "well calibrated; treat it as the default and move away from it only when the other "
    "evidence clearly and consistently disagrees. "
    "(4) Answer: p_make with three decimals. Most NBA shots are missed; long twos and "
    "threes are usually made 33-42% of the time, and only shots at the rim go much above "
    "60%. Do not infer player identities or use outside facts. Keep the rationale under "
    "80 words. " + agents.FEATURE_GUIDANCE
)
EXPECTED_OUTPUT_V1 = "JSON with a step-by-step rationale first, then p_make to three decimals"

ROLE_NAMES = {
    "effects": "Shot difficulty analyst",
    "knn": "Similar-shots analyst",
    "model": "Statistical model analyst",
}


def evidence_text(kind: str, row: dict) -> str:
    if kind == "effects":
        return row["effect_text"]
    if kind == "knn":
        k = int(row["knn_k"])
        made = int(row["knn_made"])
        return (
            f"Similar-shots retrieval: of the {k} most similar training shots (matched on "
            f"distance, defender distance, shot clock, dribbles, touch time and shot value), "
            f"{made} were made ({made / k:.1%}). The 5 closest, for context only: "
            f"{row['knn_examples']}."
        )
    if kind == "model":
        return (
            "Statistical model tool output: a gradient-boosted model trained on the same "
            "91k training shots with all nine measurements estimates p_make = "
            f"{row['p_xgboost']:.3f}"
        )
    raise ValueError(kind)


def base_rate_text(row: dict) -> str:
    """The overall and distance-band rates only (the start of the effect table)."""
    return row["effect_text"].split(" Within that distance band:")[0]


def specialist_role(kind: str, row: dict) -> str:
    """One specialist per evidence type; non-effects specialists also get the base rate."""
    parts = [f"{ROLE_NAMES[kind]}."]
    if kind != "effects":
        parts.append(base_rate_text(row))
    parts.append(evidence_text(kind, row))
    return " ".join(parts).rstrip(".")


def role(variant: str, row: dict, anchors=None, features=None, title="Shot probability analyst"):
    if variant == "V0":
        return f"{title}. {anchors.text(features)}"
    text = " ".join([f"{title}."] + [evidence_text(k, row) for k in EVIDENCE[variant]])
    return text.rstrip(".")  # CrewAI appends ". " after the role


@contextlib.contextmanager
def instructions(variant: str):
    """Swap the CrewAI goal/task text for V1+ (conditions run one at a time)."""
    saved = agents.INSTRUCTIONS, agents.EXPECTED_OUTPUT
    if variant != "V0":
        agents.INSTRUCTIONS, agents.EXPECTED_OUTPUT = INSTRUCTIONS_V1, EXPECTED_OUTPUT_V1
    try:
        yield
    finally:
        agents.INSTRUCTIONS, agents.EXPECTED_OUTPUT = saved
