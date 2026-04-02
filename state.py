from typing import TypedDict , Optional
# This is shared memory between agents
class HealthState(TypedDict , total=False):
     symptoms: str
     risk: str 
     diagnosis: str
     recommendation: str
     knowledge: list 