import random
from core.models import HospitalState, TriageAction

class RandomAgent:
    """Randomly selects a valid action each step."""

    def act(self, state: HospitalState) -> TriageAction:
        waiting = state.patients_waiting
        doctors = [d for d in state.doctors if d.status.value == "available"] or [d for d in state.doctors if d.status == "available"]

        actions = [TriageAction(action_type="wait")]

        if waiting and doctors:
            patient = random.choice(waiting)
            doctor = random.choice(doctors)
            actions.append(
                TriageAction(
                    action_type="assign",
                    patient_id=patient.id,
                    doctor_id=doctor.id,
                )
            )

        return random.choice(actions)
