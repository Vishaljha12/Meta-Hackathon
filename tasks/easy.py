from core.env import HospitalTriageEnv

class EasyTask:
    def __init__(self):
        self.description = "Low volume, steady arrivals, no surges"
    
    def make_env(self) -> HospitalTriageEnv:
        config = {
            "n_initial_patients": 5,
            "n_doctors_general": 2,
            "n_doctors_cardiac": 1,
            "n_doctors_trauma": 1,
            "n_doctors_pediatric": 1,
            "max_timesteps": 50,
            "arrival_rate": 0.3,
            "surge_enabled": False,
            "shift_changes": False
        }
        return HospitalTriageEnv(task_config=config)
