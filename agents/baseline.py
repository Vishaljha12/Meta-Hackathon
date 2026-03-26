"""
baseline.py — LLM-powered baseline agent using the OpenAI API.
This satisfies the hackathon requirement for a reproducible baseline 
that reads from the OPENAI_API_KEY environment variable.
"""

import sys
import os
import json
from openai import OpenAI

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import HospitalState, TriageAction

class LLMBaselineAgent:
    """
    An LLM-powered agent using the OpenAI API standard to solve the triage environment.
    """
    def __init__(self):
        # This automatically looks for os.environ.get("OPENAI_API_KEY")
        try:
            self.client = OpenAI() 
        except Exception:
            self.client = None
        
        self.system_prompt = """
You are an expert Chief Triage Nurse and Hospital Administrator managing a high-stakes Emergency Room. Your singular goal is to maximize patient survival and throughput.

You operate in a turn-based simulation. At each step, you will receive the current "HospitalState" in JSON format, containing a list of `patients_waiting` and `doctors`. 

You must evaluate the state and choose exactly ONE action from the following options:
1. "assign" - Assign a specific waiting patient to a specific available doctor.
2. "wait" - Do nothing and let time advance (only use this if NO doctors are available).
3. "discharge" - Discharge a patient currently being treated (rarely used, let the simulation handle discharges).

CRITICAL DECISION RULES (In order of priority):
1. CRITICAL CARE FIRST: You must immediately assign patients with severity level 1 (RESUSCITATION) and 2 (EMERGENT).
2. SPECIALTY MATCHING: Always try to match the patient's `specialty_needed` to the doctor's `specialty`.
3. DO NOT IDLE: If there are patients in the waiting room and doctors with status "available", you must make an "assign" action.

OUTPUT FORMAT:
You must output ONLY valid JSON matching the following strict schema. Do not include markdown formatting, explanations, or any other text.
{
  "action_type": "assign" | "wait" | "discharge",
  "patient_id": "string (optional)",
  "doctor_id": "string (optional)"
}
"""

    def act(self, state: HospitalState) -> TriageAction:
        # 1. Convert the Pydantic state object into a JSON string
        state_json = state.model_dump_json()

        # 2. Inject into user prompt using an f-string
        user_prompt = f"""
Current Hospital State:
{state_json}

Provide your next action in strict JSON format.
"""

        try:
            # 3. Call the API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Fast, cheap, and smart enough for this
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0 # Keep it deterministic for grading
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # 4. Clean up markdown if the LLM gets sneaky and wraps the JSON
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            # 5. Parse the JSON and return your strict OpenEnv action model
            action_dict = json.loads(raw_text.strip())
            return TriageAction(**action_dict)

        except Exception as e:
            # Fallback to waiting if the API fails, rate limits, or hallucinates bad JSON.
            # This prevents your evaluation script from crashing mid-run.
            print(f"Agent Error: {e}")
            return TriageAction(action_type="wait")