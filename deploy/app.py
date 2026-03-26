from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from core import HospitalTriageEnv
from tasks import MediumTask
from core.models import TriageAction

app = FastAPI(title="Hospital Triage OpenEnv API")

# Initialize default task environment
task = MediumTask()
env = task.make_env()

class ResetRequest(BaseModel):
    seed: int = 42

@app.post("/reset")
def reset_env(request: ResetRequest):
    state = env.reset(seed=request.seed)
    return state

@app.post("/step")
def step_env(action: TriageAction):
    try:
        result = env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Call reset() first")

@app.get("/")
def health_check():
    return {"status": "ok", "env": "hospital-triage-env", "task": "MediumTask"}
