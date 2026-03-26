"""
reward.py — The reward function. This is the HEART of your environment.

KEY CONCEPT: Partial Progress Signals
--------------------------------------
Bad reward design:  agent gets 0 until perfect, then gets 1. (Impossible to learn from)
Good reward design: agent gets small signals for EVERY correct sub-decision.

Our reward is a sum of multiple components, each 0.0 to 1.0, then weighted.

REWARD BREAKDOWN per step:
  +0.30  Treating an ESI-1/2 patient quickly (critical cases)
  +0.20  Specialty match (right doctor for condition)
  +0.20  Throughput (patients discharged this step)
  +0.15  Avoiding long waits (< 10 steps for any patient)
  -0.15  Per patient who goes CRITICAL (deteriorated while waiting)
  -0.10  Per ESI-1 patient waiting more than 3 steps (near-miss penalty)

Total range: roughly -1.0 to +1.0 per step
Episode score: sum of step rewards, normalized to 0.0-1.0 by grader
"""

from .models import (
    HospitalState, TriageAction, Patient, Doctor,
    Severity, PatientStatus, DoctorStatus, Specialty
)
from typing import Tuple, Dict


# ─── Weights (must sum to 1.0 for positive side) ───
W_CRITICAL_CARE  = 0.30
W_SPECIALTY      = 0.20
W_THROUGHPUT     = 0.20
W_WAIT_TIME      = 0.15
W_CRITICAL_MISS  = 0.15  # Penalty weight

# Thresholds
MAX_WAIT_BEFORE_PENALTY = 10   # Steps. After this, wait penalty kicks in
ESI1_MAX_WAIT           = 3    # ESI-1 should NEVER wait more than 3 steps
SPECIALTY_MATCH_BONUS   = 1.0  # Full bonus for exact match
SPECIALTY_MISMATCH_COST = 0.3  # Partial penalty if wrong specialty


def compute_step_reward(
    prev_state: HospitalState,
    action: TriageAction,
    next_state: HospitalState
) -> Tuple[float, Dict]:
    """
    Compute reward for one step transition.
    
    Returns:
        reward: float between approximately -1.0 and +1.0
        breakdown: dict explaining each component (useful for debugging)
    """
    breakdown = {}
    reward = 0.0

    # ── Component 1: Critical care priority ──────────────────────────────
    # Did we assign an ESI-1 or ESI-2 patient this step? Big reward.
    critical_care_bonus = 0.0
    if action.action_type == "assign" and action.patient_id:
        patient = _find_patient(prev_state, action.patient_id)
        if patient:
            if patient.severity == Severity.RESUSCITATION:  # ESI-1
                critical_care_bonus = 1.0   # Maximum reward
            elif patient.severity == Severity.EMERGENT:     # ESI-2
                critical_care_bonus = 0.7
            elif patient.severity == Severity.URGENT:       # ESI-3
                critical_care_bonus = 0.4
            elif patient.severity == Severity.LESS_URGENT:  # ESI-4
                critical_care_bonus = 0.2
            else:                                           # ESI-5
                critical_care_bonus = 0.1

    breakdown['critical_care'] = critical_care_bonus
    reward += W_CRITICAL_CARE * critical_care_bonus

    # ── Component 2: Specialty matching ──────────────────────────────────
    # Did we match patient's needed specialty to doctor's specialty?
    specialty_score = 0.0
    if action.action_type == "assign" and action.patient_id and action.doctor_id:
        patient = _find_patient(prev_state, action.patient_id)
        doctor = _find_doctor(prev_state, action.doctor_id)
        if patient and doctor:
            if doctor.specialty == patient.specialty_needed:
                specialty_score = SPECIALTY_MATCH_BONUS   # Perfect match
            elif doctor.specialty == Specialty.GENERAL:
                specialty_score = 0.5   # General doctor can handle anything, partial credit
            else:
                specialty_score = 0.0   # Wrong specialty

    breakdown['specialty_match'] = specialty_score
    reward += W_SPECIALTY * specialty_score

    # ── Component 3: Throughput ───────────────────────────────────────────
    # Count patients newly discharged this step
    prev_discharged = {p.id for p in prev_state.patients_being_treated
                       if p.status == PatientStatus.DISCHARGED}
    new_discharged = next_state.discharged_count - prev_state.discharged_count
    # Normalize: assume max 3 discharges per step is excellent
    throughput_score = min(new_discharged / 3.0, 1.0)

    breakdown['throughput'] = throughput_score
    reward += W_THROUGHPUT * throughput_score

    # ── Component 4: Wait time management ────────────────────────────────
    # Reward if NO patient is waiting too long
    waiting_patients = next_state.patients_waiting
    if not waiting_patients:
        wait_score = 1.0  # Nobody waiting = perfect
    else:
        over_threshold = sum(1 for p in waiting_patients
                             if p.wait_time > MAX_WAIT_BEFORE_PENALTY)
        wait_score = max(0.0, 1.0 - (over_threshold / len(waiting_patients)))

    breakdown['wait_time'] = wait_score
    reward += W_WAIT_TIME * wait_score

    # ── Penalty 1: Patients who went critical ─────────────────────────────
    new_critical = next_state.critical_count - prev_state.critical_count
    critical_penalty = new_critical * 1.0  # -1.0 per critical patient

    breakdown['critical_penalty'] = -critical_penalty
    reward -= W_CRITICAL_MISS * critical_penalty

    # ── Penalty 2: ESI-1 patients waiting too long ────────────────────────
    esi1_waiting_too_long = sum(
        1 for p in next_state.patients_waiting
        if p.severity == Severity.RESUSCITATION and p.wait_time > ESI1_MAX_WAIT
    )
    esi1_penalty = esi1_waiting_too_long * 0.5

    breakdown['esi1_near_miss'] = -esi1_penalty
    reward -= esi1_penalty

    breakdown['total'] = round(reward, 4)
    return round(reward, 4), breakdown


def _find_patient(state: HospitalState, patient_id: str) -> Patient | None:
    """Helper: find patient by ID across all queues."""
    all_patients = state.patients_waiting + state.patients_being_treated
    return next((p for p in all_patients if p.id == patient_id), None)


def _find_doctor(state: HospitalState, doctor_id: str) -> Doctor | None:
    """Helper: find doctor by ID."""
    return next((d for d in state.doctors if d.id == doctor_id), None)


def normalize_episode_score(total_reward: float, max_possible: float) -> float:
    """
    Convert raw episode reward to 0.0-1.0 score for the grader.
    
    Clamps to [0, 1] so negative episodes don't score below 0.
    """
    if max_possible <= 0:
        return 0.0
    score = total_reward / max_possible
    return max(0.0, min(1.0, score))

