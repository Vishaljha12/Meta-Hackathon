from core.env import HospitalTriageEnv

class MediumTask:
    def __init__(self):
        self.description = "Moderate volume, occasional patient surges"
    
    def make_env(self) -> HospitalTriageEnv:
        config = {
            "n_initial_patients": 10,
            "n_doctors_general": 2,
            "n_doctors_cardiac": 1,
            "n_doctors_trauma": 1,
            "n_doctors_pediatric": 1,
            "max_timesteps": 50,
            "arrival_rate": 0.5,
            "surge_enabled": True,
            "shift_changes": False
        }
        return HospitalTriageEnv(task_config=config)
