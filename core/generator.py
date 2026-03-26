"""
generator.py — Procedurally generates realistic patient scenarios.

WHY THIS MATTERS:
  The agent must generalize, not memorize.
  If every episode is identical, the agent just memorizes the answer.
  Random generation with a seed = reproducible but varied.

DESIGN:
  - Severity follows a realistic ER distribution (not 50/50 critical/non-critical)
  - Real ER stats: ~10% ESI-1/2, ~35% ESI-3, ~55% ESI-4/5
  - Complaints are matched to specialties realistically
"""

import random
from typing import List, Tuple
from .models import Patient, Doctor, Severity, Specialty, DoctorStatus

# ── Realistic severity distribution (weights) ──
# Based on real ER statistics
SEVERITY_WEIGHTS = {
    Severity.RESUSCITATION: 5,   # 5%  - life threatening
    Severity.EMERGENT:      10,  # 10% - serious
    Severity.URGENT:        35,  # 35% - needs care soon
    Severity.LESS_URGENT:   30,  # 30% - can wait
    Severity.NON_URGENT:    20,  # 20% - very minor
}

# ── Complaint → Specialty mapping ──
COMPLAINTS = {
    Specialty.CARDIAC: [
        "chest pain", "shortness of breath", "heart palpitations",
        "suspected MI", "irregular heartbeat"
    ],
    Specialty.TRAUMA: [
        "car accident injuries", "broken arm", "deep laceration",
        "fall from height", "sports injury", "stab wound"
    ],
    Specialty.PEDIATRIC: [
        "child fever 104F", "child seizure", "infant not breathing normally",
        "child abdominal pain", "pediatric asthma attack"
    ],
    Specialty.GENERAL: [
        "abdominal pain", "high fever", "urinary tract infection",
        "severe headache", "back pain", "nausea and vomiting",
        "allergic reaction", "dizziness"
    ],
}

DOCTOR_NAMES = [
    "Dr. Patel", "Dr. Singh", "Dr. Chen", "Dr. Williams",
    "Dr. Kumar", "Dr. Sharma", "Dr. Rodriguez", "Dr. Johnson",
    "Dr. Kapoor", "Dr. Li"
]

def generate_patient(arrival_time: int, seed_offset: int = 0) -> Patient:
    """Generate one realistic patient."""
    severities = list(SEVERITY_WEIGHTS.keys())
    weights    = list(SEVERITY_WEIGHTS.values())
    severity   = random.choices(severities, weights=weights, k=1)[0]

    # ESI-1 patients are almost always cardiac or trauma
    if severity == Severity.RESUSCITATION:
        specialty = random.choice([Specialty.CARDIAC, Specialty.TRAUMA])
    elif severity == Severity.EMERGENT:
        specialty = random.choice([Specialty.CARDIAC, Specialty.TRAUMA, Specialty.GENERAL])
    else:
        specialty = random.choice(list(Specialty))

    complaint = random.choice(COMPLAINTS[specialty])

    # Pediatric patients are younger
    if specialty == Specialty.PEDIATRIC:
        age = random.randint(0, 11)
    else:
        age = random.randint(12, 90)

    patient_num = arrival_time * 10 + seed_offset
    return Patient(
        name=f"Patient_{patient_num:04d}",
        age=age,
        severity=severity,
        chief_complaint=complaint,
        specialty_needed=specialty,
        arrival_time=arrival_time,
    )


def generate_patient_batch(
    n: int,
    start_time: int = 0,
    seed: int = 42
) -> List[Patient]:
    """Generate n patients for the initial queue."""
    random.seed(seed)
    return [generate_patient(start_time, i) for i in range(n)]


def generate_doctor_pool(
    n_general: int,
    n_cardiac: int = 1,
    n_trauma: int = 1,
    n_pediatric: int = 1,
    seed: int = 42
) -> List[Doctor]:
    """
    Generate a pool of doctors with different specialties.
    
    Easy task:   mostly general, 1 specialist
    Medium task: balanced mix
    Hard task:   shift-based, doctors go on break
    """
    random.seed(seed + 100)
    doctors = []
    name_pool = DOCTOR_NAMES.copy()
    random.shuffle(name_pool)
    name_idx = 0

    def make_doctor(specialty: Specialty, skill: float = 1.0) -> Doctor:
        nonlocal name_idx
        name = name_pool[name_idx % len(name_pool)]
        name_idx += 1
        return Doctor(
            name=name,
            specialty=specialty,
            skill_level=skill,
        )

    for _ in range(n_general):
        doctors.append(make_doctor(Specialty.GENERAL, skill=round(random.uniform(0.8, 1.2), 2)))
    for _ in range(n_cardiac):
        doctors.append(make_doctor(Specialty.CARDIAC, skill=round(random.uniform(1.0, 1.3), 2)))
    for _ in range(n_trauma):
        doctors.append(make_doctor(Specialty.TRAUMA, skill=round(random.uniform(1.0, 1.3), 2)))
    for _ in range(n_pediatric):
        doctors.append(make_doctor(Specialty.PEDIATRIC, skill=round(random.uniform(1.0, 1.3), 2)))

    return doctors


def generate_surge_arrivals(
    timestep: int,
    base_rate: float = 0.3,
    surge_multiplier: float = 3.0
) -> List[Patient]:
    """
    Simulate a surge event (e.g. mass casualty).
    Used in medium/hard tasks to create dynamic difficulty.
    """
    n = int(random.gauss(base_rate * surge_multiplier, 1))
    n = max(0, min(n, 5))  # Cap at 5 per step
    return [generate_patient(timestep, i) for i in range(n)]

