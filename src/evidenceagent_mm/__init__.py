"""EvidenceAgent-MM public package."""

from evidenceagent_mm.agent import EvidenceAgent
from evidenceagent_mm.schema import AgentResponse, EvidenceAtom, Modality, ResponseStatus

__all__ = ["AgentResponse", "EvidenceAgent", "EvidenceAtom", "Modality", "ResponseStatus"]
__version__ = "2.0.0"
