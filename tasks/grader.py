"""
grader.py — Scores agent performance from 0.0 to 1.0.

The grader is called after each episode ends.
It reads the EpisodeResult and returns a normalized score.

SCORING FORMULA:
  score = weighted average of sub-metrics, each 0.0 to 1.0

Sub-metrics:
  1. Treatment rate     (patients treated / total patients)       w=0.30
  2. Critical avoidance (1 - critical_rate)                       w=0.30
  3. Wait time score    (1 - avg_wait / max_acceptable_wait)      w=0.20
  4. Specialty match    (% of assignments that matched)           w=0.20
"""
from core.models import EpisodeResult

WEIGHTS = {
    "treatment_rate":    0.30,
    "critical_avoidance":0.30,
    "wait_time":         0.20,
    "specialty_match":   0.20,
}

MAX_ACCEPTABLE_WAIT = 15.0  # Steps. If avg wait > this, wait score = 0


def grade(result: EpisodeResult, total_patients_in_episode: int) -> dict:
    """
    Grade one episode. Returns dict with final score and breakdown.

    Args:
        result: EpisodeResult from env.get_episode_result()
        total_patients_in_episode: total patients that appeared (treated + waiting + critical)

    Returns:
        {
          "score": float,   # 0.0 to 1.0
          "breakdown": dict # each sub-metric score
        }
    """
    total = max(total_patients_in_episode, 1)

    # 1. Treatment rate — how many patients got treated
    treatment_rate = min(result.patients_treated / total, 1.0)

    # 2. Critical avoidance — zero critical = 1.0, all critical = 0.0
    critical_rate = result.patients_critical / total
    critical_avoidance = max(0.0, 1.0 - critical_rate * 3)  # Penalize heavily

    # 3. Wait time score
    if result.avg_wait_time <= 0:
        wait_score = 1.0
    else:
        wait_score = max(0.0, 1.0 - result.avg_wait_time / MAX_ACCEPTABLE_WAIT)

    # 4. Specialty match (already 0-1 from env)
    specialty_score = result.specialty_match_rate

    breakdown = {
        "treatment_rate":     round(treatment_rate, 3),
        "critical_avoidance": round(critical_avoidance, 3),
        "wait_time":          round(wait_score, 3),
        "specialty_match":    round(specialty_score, 3),
    }

    final_score = sum(
        breakdown[k] * WEIGHTS[k] for k in WEIGHTS
    )

    return {
        "score":     round(min(final_score, 1.0), 4),
        "breakdown": breakdown,
        "raw":       result.model_dump(),
    }
