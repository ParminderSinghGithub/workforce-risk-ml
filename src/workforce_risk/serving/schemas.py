"""Pydantic request and response schemas for the Workforce Risk Serving API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmployeePredictionRequest(BaseModel):
    """Raw employee attributes and feedback text for risk inference."""
    employee_id: Optional[str] = Field(default="EMP-00000", description="Unique employee identifier")
    department: str = Field(default="Engineering", description="Corporate department name")
    job_level: str = Field(default="Mid", description="Seniority level (Entry, Mid, Senior, Lead, Manager)")
    role: str = Field(default="Software Engineer", description="Specific job title / role")
    communication_patterns: str = Field(default="Balanced", description="Communication pattern archetype")
    persona_name: str = Field(default="Balanced Contributor", description="Employee archetype persona")

    tenure_months: float = Field(default=24.0, ge=0.0, description="Tenure in months")
    salary: float = Field(default=95000.0, ge=0.0, description="Annual base salary in USD")
    performance_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Performance rating [0, 1]")
    satisfaction_score: float = Field(default=0.70, ge=0.0, le=1.0, description="Job satisfaction rating [0, 1]")
    workload_score: float = Field(default=0.60, ge=0.0, le=1.0, description="Workload burden [0, 1]")
    team_sentiment: float = Field(default=0.70, ge=0.0, le=1.0, description="Team sentiment rating [0, 1]")
    project_completion_rate: float = Field(default=0.85, ge=0.0, le=1.0, description="Project delivery rate [0, 1]")
    overtime_hours: float = Field(default=5.0, ge=0.0, description="Weekly overtime hours")
    training_participation: float = Field(default=0.60, ge=0.0, le=1.0, description="Training completion rate [0, 1]")
    collaboration_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Cross-functional collaboration [0, 1]")
    email_sentiment: float = Field(default=0.70, ge=0.0, le=1.0, description="Email tone sentiment [0, 1]")
    slack_activity: float = Field(default=0.65, ge=0.0, le=1.0, description="Messaging activity level [0, 1]")
    meeting_participation: float = Field(default=0.70, ge=0.0, le=1.0, description="Meeting active participation [0, 1]")
    goal_achievement_rate: float = Field(default=0.80, ge=0.0, le=1.0, description="Goal fulfillment rate [0, 1]")
    stress_level: float = Field(default=0.50, ge=0.0, le=1.0, description="Reported stress index [0, 1]")
    role_complexity_score: float = Field(default=0.65, ge=0.0, le=1.0, description="Cognitive job demand [0, 1]")
    career_progression_score: float = Field(default=0.60, ge=0.0, le=1.0, description="Advancement trajectory [0, 1]")

    technical_skills: Optional[List[str]] = Field(default_factory=lambda: ["Python", "SQL", "Git"], description="Documented technical skill tags")
    soft_skills: Optional[List[str]] = Field(default_factory=lambda: ["Teamwork", "Problem Solving"], description="Documented soft skill tags")
    recent_feedback: str = Field(default="Good quarter, steady project progress and team collaboration.", description="Recent qualitative feedback text commentary")
    threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional custom decision threshold override")


class ModalityBreakdown(BaseModel):
    """Detailed attribution and log-odds contribution per modality."""
    structured_weight: float
    text_weight: float
    intercept: float
    structured_logit: float
    text_logit: float
    structured_contribution: float
    text_contribution: float


class PredictionResponse(BaseModel):
    """Calibrated multimodal prediction response."""
    employee_id: Optional[str]
    fused_risk_probability: float
    structured_risk_probability: float
    text_risk_probability: float
    risk_prediction: int
    risk_tier: str
    decision_threshold: float
    modality_breakdown: ModalityBreakdown
    summary: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    employees: List[EmployeePredictionRequest] = Field(..., min_length=1, description="List of employee records to evaluate")
    threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional decision threshold override for entire batch")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response containing multiple evaluation results."""
    total_predictions: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """API health and model readiness status response."""
    status: str
    version: str
    device: str
    models_loaded: Dict[str, bool]
    decision_threshold: float
    offline_mode: bool
