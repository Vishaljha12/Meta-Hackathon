"""
env.py — The main environment class. Implements the OpenEnv spec.

THE THREE METHODS EVERY OPENENV MUST HAVE:
  reset()          → HospitalState   (start a new episode)
  step(action)     → StepResult      (take an action, get new state + reward)
  state()          → HospitalState   (observe current state without acting)

DESIGN PRINCIPLE:
  The environment is a state machine.
  reset() initializes the state.
  step() transitions the state.
  state() is just a read-only view.
"""

from typing import Optional, List, Dict
import random
import copy

from .models import (
    HospitalState, TriageAction, StepResult, EpisodeResult,
    Patient, Doctor, PatientStatus, DoctorStatus, Severity
)
from .generator import (
    generate_patient, generate_patient_batch, generate_doctor_pool, generate_surge_arrivals
)
from .simulator import advance_simulation, apply_action_to_state
from .reward import compute_step_reward, normalize_episode_score


class HospitalTriageEnv:
    """
    Hospital Emergency Room Triage Environment.

    The agent must allocate ER patients to doctors optimally —
    prioritizing critical cases, matching specialties, and
    minimizing wait times.

    Usage:
        env = HospitalTriageEnv(task_config=EASY_CONFIG)
        state = env.reset(seed=42)

        while not state.episode_over:
            action = agent.act(state)         # Your agent decides
            result = env.step(action)
            state  = result.state

        score = env.get_episode_result()
    """

    def __init__(self, task_config: Dict):
        """
        task_config contains all task-specific parameters.
        Each task (easy/medium/hard) passes its own config.

        Config keys:
          n_initial_patients: int     — patients at episode start
          n_doctors_general:  int     — general doctors
          n_doctors_cardiac:  int     — cardiac specialists
          n_doctors_trauma:   int     — trauma specialists
          n_doctors_pediatric:int     — pediatric specialists
          max_timesteps:      int     — episode length
          arrival_rate:       float   — new patients per step
          surge_enabled:      bool    — random surge events
          shift_changes:      bool    — doctors go on break (hard only)
        """
        self.config = task_config
        self._state: Optional[HospitalState] = None
        self._episode_rewards: List[float] = []
        self._step_breakdowns: List[Dict] = []
        self._seed: Optional[int] = None

    # ─────────────────────────────────────────
    # OPENENV REQUIRED METHODS
    # ─────────────────────────────────────────

    def reset(self, seed: int = 42) -> HospitalState:
        """
        Start a new episode. Returns the initial state.

        IMPORTANT: Setting a seed makes episodes reproducible.
        Same seed = same patients = same scenario every run.
        This is required for fair evaluation.
        """
        self._seed = seed
        random.seed(seed)

        # Generate initial patients
        patients = generate_patient_batch(
            n=self.config["n_initial_patients"],
            start_time=0,
            seed=seed
        )

        # Generate doctors
        doctors = generate_doctor_pool(
            n_general=self.config["n_doctors_general"],
            n_cardiac=self.config.get("n_doctors_cardiac", 1),
            n_trauma=self.config.get("n_doctors_trauma", 1),
            n_pediatric=self.config.get("n_doctors_pediatric", 1),
            seed=seed
        )

        self._state = HospitalState(
            timestep=0,
            patients_waiting=patients,
            patients_being_treated=[],
            doctors=doctors,
            discharged_count=0,
            critical_count=0,
            total_wait_time=0.0,
            episode_over=False,
        )

        self._episode_rewards = []
        self._step_breakdowns = []
        return self._state

    def step(self, action: TriageAction) -> StepResult:
        """
        Apply action, advance simulation by 1 timestep, return result.

        Flow:
          1. Validate action
          2. Apply action to current state (e.g. assign patient → doctor)
          3. Advance time (doctors treat, patients wait, new arrivals)
          4. Compute reward
          5. Check if episode is over
          6. Return StepResult
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        prev_state = self._state

        # ── 1 & 2: Apply action ──
        mid_state, error = apply_action_to_state(prev_state, action)

        # ── 3: Advance simulation ──
        # Handle new patient arrivals
        new_arrivals = []
        if random.random() < self.config.get("arrival_rate", 0.0):
            n_arrivals = max(1, int(random.gauss(1, 0.5)))
            new_arrivals = [
                generate_patient(mid_state.timestep, i)
                for i in range(n_arrivals)
            ]

        # Surge event (medium/hard tasks)
        if self.config.get("surge_enabled", False):
            if random.random() < 0.05:  # 5% chance per step
                surge = generate_surge_arrivals(mid_state.timestep, surge_multiplier=4.0)
                new_arrivals.extend(surge)

        next_state = advance_simulation(mid_state, new_arrivals)

        # ── 4: Compute reward ──
        reward, breakdown = compute_step_reward(prev_state, action, next_state)

        # ── 5: Check episode end ──
        done = (
            next_state.timestep >= self.config["max_timesteps"]
            or (
                len(next_state.patients_waiting) == 0
                and len(next_state.patients_being_treated) == 0
                and next_state.timestep > 5  # Don't end on empty start
            )
        )

        if done:
            next_state = next_state.model_copy(update={"episode_over": True})

        # ── 6: Store and return ──
        self._state = next_state
        self._episode_rewards.append(reward)
        self._step_breakdowns.append(breakdown)

        return StepResult(
            state=next_state,
            reward=reward,
            done=done,
            info={
                "error": error,
                "reward_breakdown": breakdown,
                "timestep": next_state.timestep,
                "new_arrivals": len(new_arrivals),
            }
        )

    def state(self) -> HospitalState:
        """Return current state (read-only, no side effects)."""
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    # ─────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────

    def get_episode_result(self) -> EpisodeResult:
        """
        Called after episode ends.
        Returns structured summary for the grader.
        """
        if self._state is None:
            raise RuntimeError("No episode run yet")

        total_reward = sum(self._episode_rewards)
        max_possible = len(self._episode_rewards) * 0.85  # Rough upper bound per step

        # Count specialty match rate from breakdowns
        matches = [b.get('specialty_match', 0) for b in self._step_breakdowns]
        match_rate = sum(matches) / max(len(matches), 1)

        # Count ESI-1 patients who went critical
        # (We track these separately in the simulator)
        severity1_missed = self._state.critical_count  # Simplified for now

        avg_wait = (
            self._state.total_wait_time /
            max(self._state.discharged_count + len(self._state.patients_being_treated), 1)
        )

        return EpisodeResult(
            total_reward=round(total_reward, 4),
            patients_treated=self._state.discharged_count,
            patients_critical=self._state.critical_count,
            avg_wait_time=round(avg_wait, 2),
            severity1_missed=severity1_missed,
            specialty_match_rate=round(match_rate, 3),
            timesteps_taken=self._state.timestep,
        )

    def action_space(self) -> Dict:
        """
        Return valid actions for current state.
        Useful for agents to know what moves are legal.
        """
        if self._state is None:
            return {}

        available_doctors = [
            d for d in self._state.doctors if d.status == DoctorStatus.AVAILABLE
        ]
        waiting_patients = self._state.patients_waiting

        valid_assigns = [
            {"action_type": "assign", "patient_id": p.id, "doctor_id": d.id}
            for p in waiting_patients
            for d in available_doctors
        ]

        return {
            "assign": valid_assigns,
            "wait":   [{"action_type": "wait"}],
            "discharge": [
                {"action_type": "discharge", "patient_id": p.id}
                for p in self._state.patients_being_treated
            ]
        }

