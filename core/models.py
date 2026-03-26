"""
models.py — All typed data models for the Hospital Triage Environment.

WHY PYDANTIC?
- Validates data automatically (e.g. severity must be 1-5, not 99)
- Converts to/from JSON easily (needed for the OpenEnv API)
- Gives you autocomplete in your editor
- Makes bugs obvious early (wrong type = immediate error)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum
import uuid
from datetime import datetime


# ─────────────────────────────────────────────
# ENUMS — Fixed categories, not free text
# ─────────────────────────────────────────────

class Severity(int, Enum):
    """
    ESI (Emergency Severity Index) — the real triage scale used in hospitals.
    1 = Resuscitation (dying NOW)
    2 = Emergent (could die in minutes)
    3 = Urgent (needs care within 30 min)
    4 = Less Urgent (can wait 1-2 hours)
    5 = Non-Urgent (can wait many hours)
    """
    RESUSCITATION = 1
    EMERGENT      = 2
    URGENT        = 3
    LESS_URGENT   = 4
    NON_URGENT    = 5


class PatientStatus(str, Enum):
    WAITING    = "waiting"     # In queue, not yet assigned
    ASSIGNED   = "assigned"    # Being treated by a doctor
    DISCHARGED = "discharged"  # Treatment done, gone home
    CRITICAL   = "critical"    # Deteriorated while waiting — BAD outcome


class DoctorStatus(str, Enum):
    AVAILABLE = "available"  # Free to take a patient
    BUSY      = "busy"       # Currently treating someone
    ON_BREAK  = "on_break"   # Not available (shift rotation in hard task)


class Specialty(str, Enum):
    """
    Doctors have specialties. Matching patient to right specialty = bonus reward.
    Mismatch doesn't fail — just lower reward. Partial credit!
    """
    GENERAL   = "general"    # Handles anything (slower)
    CARDIAC   = "cardiac"    # Heart attacks, chest pain
    TRAUMA    = "trauma"     # Accidents, injuries
    PEDIATRIC = "pediatric"  # Children under 12


# ─────────────────────────────────────────────
# CORE ENTITIES
# ─────────────────────────────────────────────

class Patient(BaseModel):
    """
    A single patient in the ER waiting room.
    All fields are observable by the agent (the agent can "see" these).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                          # e.g. "Patient_042" (anonymized)
    age: int = Field(ge=0, le=120)     # ge = greater-or-equal, le = less-or-equal
    severity: Severity                 # 1-5 ESI scale
    chief_complaint: str               # e.g. "chest pain", "broken arm"
    specialty_needed: Specialty        # What type of doctor is best
    arrival_time: int                  # Simulation timestep when they arrived
    wait_time: int = 0                 # How many steps they've been waiting
    status: PatientStatus = PatientStatus.WAITING
    assigned_doctor_id: Optional[str] = None

    # Hidden deterioration counter — increases each step while waiting
    # Agent cannot see this directly, but sees symptoms worsen
    deterioration_score: float = 0.0

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v


class Doctor(BaseModel):
    """
    A doctor available in the ER.
    The agent assigns patients to doctors — this is the core action space.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                         # e.g. "Dr. Smith"
    specialty: Specialty
    status: DoctorStatus = DoctorStatus.AVAILABLE
    current_patient_id: Optional[str] = None
    treatment_time_remaining: int = 0  # Steps until they finish current patient
    patients_treated: int = 0         # Track workload
    skill_level: float = Field(default=1.0, ge=0.5, le=1.5)  # Multiplier on speed


class TriageAction(BaseModel):
    """
    The ACTION the agent takes each step.
    
    The agent picks: which patient gets assigned to which doctor.
    It can also choose to WAIT (do nothing) or DISCHARGE a patient early.
    
    This is what step(action) receives.
    """
    action_type: str  # "assign", "wait", "discharge"
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None

    @field_validator('action_type')
    def validate_action_type(cls, v):
        valid = {"assign", "wait", "discharge"}
        if v not in valid:
            raise ValueError(f'action_type must be one of {valid}')
        return v


class HospitalState(BaseModel):
    """
    The STATE returned by state() and reset().
    
    This is everything the agent can observe. Think of it as the
    agent "looking at the ER whiteboard."
    
    The OpenEnv spec requires state() to return a typed object.
    """
    timestep: int                         # Current simulation time
    patients_waiting: List[Patient]       # Queue of unassigned patients
    patients_being_treated: List[Patient] # Currently with doctors
    doctors: List[Doctor]                 # All doctors and their status
    discharged_count: int = 0            # How many patients successfully treated
    critical_count: int = 0             # How many deteriorated (bad outcomes)
    total_wait_time: float = 0.0        # Running avg wait time
    episode_over: bool = False


class StepResult(BaseModel):
    """
    What step(action) returns.
    Standard OpenEnv spec: (state, reward, done, info)
    """
    state: HospitalState
    reward: float          # -1.0 to +1.0 per step
    done: bool             # Is episode finished?
    info: Dict             # Extra debug info (not used by agent, helps debugging)


class EpisodeResult(BaseModel):
    """
    Final summary after an episode ends.
    The grader reads this to compute 0.0-1.0 score.
    """
    total_reward: float
    patients_treated: int
    patients_critical: int
    avg_wait_time: float
    severity1_missed: int   # ESI-1 patients who went critical — worst failure
    specialty_match_rate: float  # % of assignments that matched specialty
    timesteps_taken: int

