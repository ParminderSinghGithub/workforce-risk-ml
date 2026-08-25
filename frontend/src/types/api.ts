export type RiskTier = 'LOW' | 'ELEVATED' | 'HIGH' | 'CRITICAL';

export interface EmployeePredictionRequest {
  employee_id?: string;
  department: string;
  job_level: string;
  role: string;
  communication_patterns: string;
  persona_name: string;
  tenure_months: number;
  salary: number;
  performance_score: number;
  satisfaction_score: number;
  workload_score: number;
  team_sentiment: number;
  project_completion_rate: number;
  overtime_hours: number;
  training_participation: number;
  collaboration_score: number;
  email_sentiment: number;
  slack_activity: number;
  meeting_participation: number;
  goal_achievement_rate: number;
  stress_level: number;
  role_complexity_score: number;
  career_progression_score: number;
  technical_skills?: string[];
  soft_skills?: string[];
  recent_feedback: string;
  threshold_override?: number | null;
}

export interface ModalityBreakdown {
  structured_weight: number;
  text_weight: number;
  intercept: number;
  structured_logit: number;
  text_logit: number;
  structured_contribution: number;
  text_contribution: number;
}

export interface PredictionResponse {
  employee_id: string;
  fused_risk_probability: number;
  structured_risk_probability: number;
  text_risk_probability: number;
  risk_prediction: number;
  risk_tier: RiskTier;
  decision_threshold: number;
  modality_breakdown: ModalityBreakdown;
  summary: string;
}

export interface BatchPredictionRequest {
  employees: EmployeePredictionRequest[];
  threshold_override?: number | null;
}

export interface BatchPredictionResponse {
  total_predictions: number;
  predictions: PredictionResponse[];
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_loaded: {
    structured_mlp: boolean;
    text_distilbert_lora: boolean;
    multimodal_late_fusion: boolean;
  };
  decision_threshold: number;
  offline_mode: boolean;
}

export interface ModelInfoResponse {
  platform: string;
  version: string;
  architecture: {
    structured_branch: string;
    text_branch: string;
    fusion_mechanism: string;
  };
  evaluation_benchmarks: {
    optimal_thresholds?: Record<string, number>;
    model_comparison_holdout_test?: Record<string, any>;
    coefficients?: Record<string, any>;
  };
}
