from core.env import HospitalTriageEnv

class HardTask:
    def __init__(self):
        self.description = "High volume, frequent surges, shift changes"
    
    def make_env(self) -> HospitalTriageEnv:
        config = {
            "n_initial_patients": 15,
            "n_doctors_general": 1,
            "n_doctors_cardiac": 1,
            "n_doctors_trauma": 1,
            "n_doctors_pediatric": 1,
            "max_timesteps": 50,
            "arrival_rate": 0.8,
            "surge_enabled": True,
            "shift_changes": True
        }
        return HospitalTriageEnv(task_config=config)
