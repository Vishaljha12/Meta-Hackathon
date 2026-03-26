"""
run_eval.py — The official evaluation script. Run this to get reproducible scores.

Usage:
    python agents/run_eval.py

Output: scores for all 3 tasks, both agents, across 5 seeds.
This is what judges will run to verify your environment works.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import HospitalTriageEnv
from tasks import EasyTask, MediumTask, HardTask, grade
from agents.baseline import LLMBaselineAgent
from agents.random_agent import RandomAgent

SEEDS = [42, 123, 7, 999, 2024]


def run_episode(env: HospitalTriageEnv, agent, seed: int) -> dict:
    """Run one complete episode and return graded result."""
    state = env.reset(seed=seed)

    total_patients_seen = len(state.patients_waiting)

    while not state.episode_over:
        action = agent.act(state)
        result = env.step(action)
        state  = result.state
        total_patients_seen = max(
            total_patients_seen,
            len(state.patients_waiting) + len(state.patients_being_treated) + state.discharged_count
        )

    episode = env.get_episode_result()
    graded  = grade(episode, total_patients_seen)
    return graded


def run_all_evals():
    print("=" * 60)
    print("  Hospital Triage OpenEnv — Evaluation Results")
    print("=" * 60)

    tasks = [
        ("Easy",   EasyTask()),
        ("Medium", MediumTask()),
        ("Hard",   HardTask()),
    ]
    agents = [
        ("LLM-Baseline", LLMBaselineAgent()),
        ("Random",     RandomAgent()),
    ]

    for task_name, task in tasks:
        env = task.make_env()
        print(f"\n  Task: {task_name} — {task.description}")
        print(f"  {'-'*50}")

        for agent_name, agent in agents:
            scores = []
            for seed in SEEDS:
                result = run_episode(env, agent, seed)
                scores.append(result["score"])

            avg = sum(scores) / len(scores)
            mn  = min(scores)
            mx  = max(scores)
            print(f"  {agent_name:15s}  avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}")
            print(f"  {'':15s}  scores: {[round(s,3) for s in scores]}")

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_evals()
