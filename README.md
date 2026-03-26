# Hospital Triage OpenEnv

## Environment Description & Motivation

This environment simulates a high-stakes Hospital Emergency Room Triage system. The motivation is to provide an evaluation testbed for AI agents to balance complex constraints under pressure. Agents act as the Chief Triage Nurse and Administrator, tasked with allocating limited resources (doctors) to continuously arriving patients based on emergency severity (ESI scale) and specialty requirements to maximize throughput and minimize wait-time deterioration.

## Action & Observation Spaces (List your JSON structure here)

### Observation Space (HospitalState)
```json
{
  "timestep": 0,
  "patients_waiting": [
    {
      "id": "e8a9b1c2",
      "name": "Patient_042",
      "age": 45,
      "severity": 1,
      "chief_complaint": "chest pain",
      "specialty_needed": "cardiac",
      "arrival_time": 0,
      "wait_time": 0,
      "status": "waiting",
      "assigned_doctor_id": null
    }
  ],
  "patients_being_treated": [],
  "doctors": [
    {
      "id": "f5d6e2b1",
      "name": "Dr. Smith",
      "specialty": "general",
      "status": "available",
      "current_patient_id": null,
      "treatment_time_remaining": 0,
      "patients_treated": 0,
      "skill_level": 1.0
    }
  ],
  "discharged_count": 0,
  "critical_count": 0,
  "total_wait_time": 0.0,
  "episode_over": false
}
```

### Action Space (TriageAction)
```json
{
  "action_type": "assign",
  "patient_id": "e8a9b1c2",
  "doctor_id": "f5d6e2b1"
}
```
*Valid `action_type` values:* `"assign"`, `"wait"`, `"discharge"`.

## Tasks (Easy, Medium, Hard)

- **Easy**: Low patient volume, steady, predictable arrivals, and no sudden surges. (Expected score range: 0.55 - 0.75)
- **Medium**: Moderate patient volume with occasional random patient surges that require careful prioritization. (Expected score range: 0.45 - 0.65)
- **Hard**: High patient volume, frequent dangerous surges, and dynamic shift changes where doctors unexpectedly go on break. (Expected score range: 0.35 - 0.55)

## Setup & Usage Instructions

1. **Install Dependencies** (If using natively, without `uv`):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Local Server** (using OpenEnv hackathon `uv` command):
   ```bash
   uv run server
   ```
   *This starts the environment server locally as defined in `pyproject.toml`.*
3. **Run Evaluation (Locally)**:
   Ensure your `OPENAI_API_KEY` is exported for the LLM baseline to work. Then run the evaluation script:
   ```bash
   python agents/run_eval.py
   ```

## Baseline Scores 

============================================================
  Hospital Triage OpenEnv — Evaluation Results
============================================================

  Task: Easy — Low volume, steady arrivals, no surges
  --------------------------------------------------
  LLM-Baseline     avg=0.867  min=0.857  max=0.873
                   scores: [0.873, 0.857, 0.866, 0.872, 0.868]
  Random           avg=0.716  min=0.506  max=0.802
                   scores: [0.766, 0.747, 0.76, 0.802, 0.506]

  Task: Medium — Moderate volume, occasional patient surges
  --------------------------------------------------
  LLM-Baseline     avg=0.861  min=0.812  max=0.909
                   scores: [0.884, 0.864, 0.812, 0.835, 0.909]
  Random           avg=0.453  min=0.358  max=0.562
                   scores: [0.436, 0.403, 0.562, 0.358, 0.505]

  Task: Hard — High volume, frequent surges, shift changes
  --------------------------------------------------
  LLM-Baseline     avg=0.636  min=0.614  max=0.651
                   scores: [0.631, 0.636, 0.646, 0.614, 0.651]
  Random           avg=0.350  min=0.310  max=0.430
                   scores: [0.43, 0.31, 0.346, 0.313, 0.351]

============================================================
  Evaluation complete.
============================================================
