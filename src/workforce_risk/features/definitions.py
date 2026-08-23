"""Feature definitions, allowlists, leakage exclusions, and manifest generator."""

from typing import Any, Dict, List

# Explicitly excluded columns from structured model features to prevent target/data leakage
EXCLUDED_LEAKAGE_COLUMNS = [
    "employee_id",
    "turnover_reason",
    "turnover_probability_generated",
    "risk_factors_summary",
    "burnout_risk",
    "left_company",
    "recent_feedback",
]

# Allowlisted base raw predictor columns
ALLOWLISTED_PREDICTORS = [
    # Categorical predictors
    "department",
    "job_level",
    "role",
    "communication_patterns",
    "persona_name",
    # Numerical predictors
    "tenure_months",
    "salary",
    "performance_score",
    "satisfaction_score",
    "workload_score",
    "team_sentiment",
    "project_completion_rate",
    "overtime_hours",
    "training_participation",
    "collaboration_score",
    "email_sentiment",
    "slack_activity",
    "meeting_participation",
    "goal_achievement_rate",
    "stress_level",
    "role_complexity_score",
    "career_progression_score",
    # Array fields
    "technical_skills",
    "soft_skills",
]

# Structured feature dictionary metadata
FEATURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Raw Continuous Numerical Features
    "tenure_months": {
        "source": ["tenure_months"],
        "type": "numeric",
        "transformation": "raw_integer_to_double",
        "description": "Total employee tenure in months",
    },
    "salary": {
        "source": ["salary"],
        "type": "numeric",
        "transformation": "raw_continuous_compensation",
        "description": "Annual salary in USD",
    },
    "performance_score": {
        "source": ["performance_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Performance evaluation score [0, 1]",
    },
    "satisfaction_score": {
        "source": ["satisfaction_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Employee reported satisfaction score [0, 1]",
    },
    "workload_score": {
        "source": ["workload_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Reported workload burden score [0, 1]",
    },
    "team_sentiment": {
        "source": ["team_sentiment"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Peer and team collaboration sentiment score [0, 1]",
    },
    "project_completion_rate": {
        "source": ["project_completion_rate"],
        "type": "numeric",
        "transformation": "raw_rate_continuous",
        "description": "Historical project delivery completion rate [0, 1]",
    },
    "overtime_hours": {
        "source": ["overtime_hours"],
        "type": "numeric",
        "transformation": "raw_hours_continuous",
        "description": "Weekly overtime hours",
    },
    "training_participation": {
        "source": ["training_participation"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Internal training and upskilling participation rate [0, 1]",
    },
    "collaboration_score": {
        "source": ["collaboration_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Cross-functional collaboration rating [0, 1]",
    },
    "email_sentiment": {
        "source": ["email_sentiment"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Aggregated email communication tone sentiment [0, 1]",
    },
    "slack_activity": {
        "source": ["slack_activity"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Normalized messaging and interaction frequency [0, 1]",
    },
    "meeting_participation": {
        "source": ["meeting_participation"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Meeting attendance and active contribution score [0, 1]",
    },
    "goal_achievement_rate": {
        "source": ["goal_achievement_rate"],
        "type": "numeric",
        "transformation": "raw_rate_continuous",
        "description": "Quarterly target and KPI goal fulfillment rate [0, 1]",
    },
    "stress_level": {
        "source": ["stress_level"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Reported physiological and mental stress index [0, 1]",
    },
    "role_complexity_score": {
        "source": ["role_complexity_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Objective job complexity and cognitive demand score [0, 1]",
    },
    "career_progression_score": {
        "source": ["career_progression_score"],
        "type": "numeric",
        "transformation": "raw_score_continuous",
        "description": "Internal mobility and advancement trajectory score [0, 1]",
    },
    # Derived Numerical Features
    "num_technical_skills": {
        "source": ["technical_skills"],
        "type": "numeric",
        "transformation": "array_size(technical_skills)",
        "description": "Total count of documented technical skill competencies",
    },
    "num_soft_skills": {
        "source": ["soft_skills"],
        "type": "numeric",
        "transformation": "array_size(soft_skills)",
        "description": "Total count of documented interpersonal competencies",
    },
    "tenure_years": {
        "source": ["tenure_months"],
        "type": "numeric",
        "transformation": "tenure_months / 12.0",
        "description": "Calculated organizational tenure in decimal years",
    },
    "workload_stress_interaction": {
        "source": ["workload_score", "stress_level"],
        "type": "numeric",
        "transformation": "workload_score * stress_level",
        "description": "Multiplicative interaction measuring compounding burnout pressure",
    },
    "satisfaction_workload_gap": {
        "source": ["satisfaction_score", "workload_score"],
        "type": "numeric",
        "transformation": "satisfaction_score - workload_score",
        "description": "Differential measuring friction between job sentiment and workload demand",
    },
    "overtime_intensity": {
        "source": ["overtime_hours"],
        "type": "numeric",
        "transformation": "overtime_hours / (overtime_hours + 40.0)",
        "description": "Bounded non-linear overtime burden ratio in [0, 1)",
    },
    "engagement_score": {
        "source": ["slack_activity", "meeting_participation", "collaboration_score"],
        "type": "numeric",
        "transformation": "(slack_activity + meeting_participation + collaboration_score) / 3.0",
        "description": "Composite index reflecting multi-channel workplace engagement",
    },
    # Categorical Indexed Features
    "department_idx": {
        "source": ["department"],
        "type": "categorical",
        "transformation": "string_indexer(department)",
        "description": "Ordinal integer index for corporate department",
    },
    "job_level_idx": {
        "source": ["job_level"],
        "type": "categorical",
        "transformation": "string_indexer(job_level)",
        "description": "Ordinal integer index for seniority level",
    },
    "role_idx": {
        "source": ["role"],
        "type": "categorical",
        "transformation": "string_indexer(role)",
        "description": "Ordinal integer index for professional job title",
    },
    "communication_patterns_idx": {
        "source": ["communication_patterns"],
        "type": "categorical",
        "transformation": "string_indexer(communication_patterns)",
        "description": "Ordinal integer index for primary communication style",
    },
    "persona_name_idx": {
        "source": ["persona_name"],
        "type": "categorical",
        "transformation": "string_indexer(persona_name)",
        "description": "Ordinal integer index for archetype persona cluster",
    },
}


def generate_feature_manifest() -> Dict[str, Any]:
    """Generate a machine-readable feature manifest describing all features and leakage controls."""
    feature_list = []
    for fname, meta in FEATURE_DEFINITIONS.items():
        feature_list.append({
            "feature_name": fname,
            "source_columns": meta["source"],
            "feature_type": meta["type"],
            "transformation": meta["transformation"],
            "description": meta["description"],
            "used_by_structured_model": True,
        })

    return {
        "total_structured_features": len(feature_list),
        "feature_count_by_type": {
            "numeric": sum(1 for f in feature_list if f["feature_type"] == "numeric"),
            "categorical": sum(1 for f in feature_list if f["feature_type"] == "categorical"),
        },
        "features": feature_list,
        "excluded_columns": {
            "columns": EXCLUDED_LEAKAGE_COLUMNS,
            "count": len(EXCLUDED_LEAKAGE_COLUMNS),
            "rationale": "Mandatory leakage exclusions, target variables, identifiers, and cross-modality text field.",
        },
        "target_variable": {
            "name": "left_company",
            "type": "binary_integer",
            "description": "Primary prediction target (1 = left company, 0 = retained)",
        },
    }
