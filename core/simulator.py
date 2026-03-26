"""
simulator.py — Simulates time passing in the ER.

Each call to step() advances the simulation by 1 timestep (~5 minutes real time).

WHAT HAPPENS EACH TIMESTEP:
  1. Doctors make progress on current patients (treatment_time_remaining -=1)
  2. Patients in treatment get discharged when timer hits 0
  3. Waiting patients accumulate wait_time
  4. High-severity patients deteriorate if waiting too long
  5. New patients may arrive (in medium/hard tasks)
"""

import random
from typing import List, Tuple
from .models import (
    HospitalState, Patient, Doctor,
    Severity, PatientStatus, DoctorStatus
)

# Deterioration thresholds — steps before each severity starts to worsen
DETERIORATION_THRESHOLDS = {
    Severity.RESUSCITATION: 2,   # ESI-1: deteriorates after 2 steps — very fast
    Severity.EMERGENT:      5,   # ESI-2: after 5 steps
    Severity.URGENT:        12,  # ESI-3: after 12 steps
    Severity.LESS_URGENT:   25,  # ESI-4: after 25 steps
    Severity.NON_URGENT:    50,  # ESI-5: takes very long
}

# Treatment time (in timesteps) by severity
# Sicker patients need MORE treatment time
TREATMENT_TIMES = {
    Severity.RESUSCITATION: random.randint,  # See below
    Severity.EMERGENT:      (8, 12),
    Severity.URGENT:        (5, 8),
    Severity.LESS_URGENT:   (2, 5),
    Severity.NON_URGENT:    (1, 3),
}

TREATMENT_TIME_RANGES = {
    Severity.RESUSCITATION: (12, 20),
    Severity.EMERGENT:      (8,  12),
    Severity.URGENT:        (5,  8),
    Severity.LESS_URGENT:   (2,  5),
    Severity.NON_URGENT:    (1,  3),
}


def get_treatment_time(severity: Severity, skill_level: float = 1.0) -> int:
    """How long this patient takes to treat (adjusted for doctor skill)."""
    lo, hi = TREATMENT_TIME_RANGES[severity]
    base = random.randint(lo, hi)
    # Skilled doctors treat faster
    adjusted = int(base / skill_level)
    return max(1, adjusted)


def advance_simulation(state: HospitalState, new_arrivals: List[Patient] = None) -> HospitalState:
    """
    Advance the simulation by 1 timestep WITHOUT applying an action.
    Called internally by env.step() after the action is applied.

    Returns updated state.
    """
    # Work on mutable copies
    doctors = [d.model_copy() for d in state.doctors]
    being_treated = [p.model_copy() for p in state.patients_being_treated]
    waiting = [p.model_copy() for p in state.patients_waiting]
    discharged_count = state.discharged_count
    critical_count = state.critical_count
    total_wait_time = state.total_wait_time

    newly_discharged = []
    still_in_treatment = []

    # ── Step 1: Advance treatment timers ──
    for patient in being_treated:
        doctor = next((d for d in doctors if d.id == patient.assigned_doctor_id), None)
        if doctor:
            doctor.treatment_time_remaining = max(0, doctor.treatment_time_remaining - 1)
            if doctor.treatment_time_remaining == 0:
                # Patient is done!
                patient.status = PatientStatus.DISCHARGED
                doctor.status = DoctorStatus.AVAILABLE
                doctor.current_patient_id = None
                doctor.patients_treated += 1
                newly_discharged.append(patient)
                discharged_count += 1
            else:
                still_in_treatment.append(patient)
        else:
            still_in_treatment.append(patient)

    # ── Step 2: Aging waiting patients ──
    still_waiting = []
    for patient in waiting:
        patient.wait_time += 1
        total_wait_time += 1

        # Deterioration check
        threshold = DETERIORATION_THRESHOLDS.get(patient.severity, 50)
        if patient.wait_time > threshold:
            patient.deterioration_score += 0.1 * (patient.severity.value == 1 and 0.5 or 0.1)
            # ESI-1 deteriorates fast; others slow
            if patient.severity in (Severity.RESUSCITATION, Severity.EMERGENT):
                patient.deterioration_score += 0.15

        # If deterioration is severe enough, patient goes critical
        if patient.deterioration_score >= 1.0:
            patient.status = PatientStatus.CRITICAL
            critical_count += 1
            # Don't add to still_waiting — they're removed (transferred to ICU)
        else:
            still_waiting.append(patient)

    # ── Step 3: Add new arrivals ──
    if new_arrivals:
        still_waiting.extend(new_arrivals)

    # ── Step 4: Sort waiting list by severity (natural ER triage order) ──
    # This means the agent sees the most critical patients first
    still_waiting.sort(key=lambda p: (p.severity.value, p.wait_time * -1))

    return HospitalState(
        timestep=state.timestep + 1,
        patients_waiting=still_waiting,
        patients_being_treated=still_in_treatment,
        doctors=doctors,
        discharged_count=discharged_count,
        critical_count=critical_count,
        total_wait_time=total_wait_time,
        episode_over=state.episode_over,
    )


def apply_action_to_state(state: HospitalState, action) -> Tuple[HospitalState, str]:
    """
    Apply the agent's action to the state BEFORE advancing time.

    Returns: (new_state, error_message_or_empty)
    """
    from .models import TriageAction
    doctors = [d.model_copy() for d in state.doctors]
    waiting = [p.model_copy() for p in state.patients_waiting]
    being_treated = [p.model_copy() for p in state.patients_being_treated]

    error = ""

    if action.action_type == "assign":
        patient = next((p for p in waiting if p.id == action.patient_id), None)
        doctor  = next((d for d in doctors if d.id == action.doctor_id), None)

        if not patient:
            error = f"Patient {action.patient_id} not found in waiting list"
        elif not doctor:
            error = f"Doctor {action.doctor_id} not found"
        elif doctor.status != DoctorStatus.AVAILABLE:
            error = f"Doctor {action.doctor_id} is not available"
        else:
            # Valid assignment
            patient.status = PatientStatus.ASSIGNED
            patient.assigned_doctor_id = doctor.id
            doctor.status = DoctorStatus.BUSY
            doctor.current_patient_id = patient.id
            doctor.treatment_time_remaining = get_treatment_time(
                patient.severity, doctor.skill_level
            )
            waiting = [p for p in waiting if p.id != patient.id]
            being_treated.append(patient)

    elif action.action_type == "discharge":
        # Early discharge (for non-urgent patients to free up space)
        patient = next((p for p in being_treated if p.id == action.patient_id), None)
        if patient:
            doctor = next((d for d in doctors if d.id == patient.assigned_doctor_id), None)
            patient.status = PatientStatus.DISCHARGED
            if doctor:
                doctor.status = DoctorStatus.AVAILABLE
                doctor.current_patient_id = None
            being_treated = [p for p in being_treated if p.id != patient.id]

    # "wait" action — do nothing this step

    new_state = HospitalState(
        timestep=state.timestep,
        patients_waiting=waiting,
        patients_being_treated=being_treated,
        doctors=doctors,
        discharged_count=state.discharged_count,
        critical_count=state.critical_count,
        total_wait_time=state.total_wait_time,
        episode_over=state.episode_over,
    )
    return new_state, error

