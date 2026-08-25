"""Data schemas and typed models for workforce risk inference."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskTier(str, Enum):
    """Categorical risk severity tiers calibrated against enterprise attrition distributions."""
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_probability(cls, prob: float, threshold: float = 0.22) -> "RiskTier":
        """Map continuous fused exit probability to risk tier."""
        if prob < threshold * 0.8:
            return cls.LOW
        elif prob < threshold * 1.5:
            return cls.ELEVATED
        elif prob < 0.60:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class EmployeeInput:
    """Raw employee attributes and feedback text provided for risk inference."""
    # Identification
    employee_id: Optional[str] = "EMP-00000"

    # Categorical Attributes (Strings or Ordinal Indices)
    department: str = "Engineering"
    job_level: str = "Mid"
    role: str = "Software Engineer"
    communication_patterns: str = "Balanced"
    persona_name: str = "Balanced Contributor"

    # Continuous Numerical Metrics
    tenure_months: float = 24.0
    salary: float = 95000.0
    performance_score: float = 0.75
    satisfaction_score: float = 0.70
    workload_score: float = 0.60
    team_sentiment: float = 0.70
    project_completion_rate: float = 0.85
    overtime_hours: float = 5.0
    training_participation: float = 0.60
    collaboration_score: float = 0.75
    email_sentiment: float = 0.70
    slack_activity: float = 0.65
    meeting_participation: float = 0.70
    goal_achievement_rate: float = 0.80
    stress_level: float = 0.50
    role_complexity_score: float = 0.65
    career_progression_score: float = 0.60

    # Skills lists
    technical_skills: List[str] = field(default_factory=lambda: ["Python", "SQL", "Git"])
    soft_skills: List[str] = field(default_factory=lambda: ["Teamwork", "Problem Solving"])

    # Unstructured Text Feedback
    recent_feedback: str = "Good quarter, steady project progress and team collaboration."

    # Direct index overrides (optional, for pre-indexed inputs)
    department_idx: Optional[int] = None
    job_level_idx: Optional[int] = None
    role_idx: Optional[int] = None
    communication_patterns_idx: Optional[int] = None
    persona_name_idx: Optional[int] = None
    num_technical_skills: Optional[float] = None
    num_soft_skills: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert input data to standard dictionary."""
        return asdict(self)


@dataclass
class RiskPredictionResult:
    """Complete multimodal risk inference evaluation result."""
    employee_id: Optional[str]
    fused_risk_probability: float
    structured_risk_probability: float
    text_risk_probability: float
    risk_prediction: int
    risk_tier: str
    decision_threshold: float
    modality_breakdown: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return asdict(self)
