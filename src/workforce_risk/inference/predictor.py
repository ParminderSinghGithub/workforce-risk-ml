"""Production-grade multimodal inference pipeline for workforce attrition risk prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from workforce_risk.fusion.model import MultimodalLateFusion, load_fusion_model, safe_logit
from workforce_risk.inference.mappings import CATEGORICAL_MAPPINGS
from workforce_risk.inference.schemas import EmployeeInput, RiskPredictionResult, RiskTier
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.preprocessor import TabularPreprocessor
from workforce_risk.nlp.model import load_lora_text_model


class WorkforceRiskPredictor:
    """End-to-end multimodal risk predictor operating strictly offline with saved model artifacts."""

    def __init__(
        self,
        structured_model: StructuredMLP,
        preprocessor: TabularPreprocessor,
        text_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        fusion_model: MultimodalLateFusion,
        categorical_mappings: Optional[Dict[str, Dict[str, int]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.structured_model = structured_model
        self.preprocessor = preprocessor
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.fusion_model = fusion_model
        self.categorical_mappings = categorical_mappings or CATEGORICAL_MAPPINGS
        self.device = device or torch.device("cpu")

        self.structured_model.eval()
        self.text_model.eval()

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Optional[str | Path] = None,
        device_str: str = "cpu",
    ) -> "WorkforceRiskPredictor":
        """Load and instantiate the complete multimodal predictor from artifact directories.

        Args:
            artifacts_dir: Root directory containing unimodal and fusion artifacts.
            device_str: Device identifier ("cpu" or "cuda").
        """
        root = Path(artifacts_dir or "artifacts").resolve()
        struct_ckpt_path = root / "structured_model" / "best_checkpoint.pt"
        text_model_dir = root / "text_transformer" / "best_model"
        fusion_model_path = root / "fusion" / "fusion_model.joblib"

        if not struct_ckpt_path.exists() or not text_model_dir.exists() or not fusion_model_path.exists():
            repo_id = os.environ.get("HF_MODEL_REPO_ID", "ParminderzHuggingFace/sentinel-workforce-risk-models")
            if repo_id:
                try:
                    from huggingface_hub import snapshot_download
                    print(f"[Model Loader] Local artifacts missing at '{root}'. Fetching inference package from Hugging Face '{repo_id}'...")
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(root),
                        allow_patterns=["structured_model/*", "text_transformer/*", "fusion/*", "evaluation_summary.json"]
                    )
                except Exception as e:
                    print(f"[Model Loader Warning] Failed to download from Hugging Face '{repo_id}': {e}")

        if not struct_ckpt_path.exists():
            raise FileNotFoundError(f"Missing structured checkpoint: {struct_ckpt_path}")
        if not text_model_dir.exists():
            raise FileNotFoundError(f"Missing text model dir: {text_model_dir}")
        if not fusion_model_path.exists():
            raise FileNotFoundError(f"Missing fusion model: {fusion_model_path}")

        # Select Device
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # 1. Load Structured MLP and Preprocessor
        saved_struct = torch.load(struct_ckpt_path, map_location=device, weights_only=False)
        preprocessor = TabularPreprocessor.from_dict(saved_struct["preprocessing_config"])
        structured_model = StructuredMLP(
            input_dim=saved_struct["model_config"]["input_dim"],
            hidden_dims=saved_struct["model_config"]["hidden_dims"],
            dropout=saved_struct["model_config"]["dropout"],
        ).to(device)
        structured_model.load_state_dict(saved_struct["model_state_dict"])
        structured_model.eval()

        # 2. Load DistilBERT + LoRA Text Model and Tokenizer
        text_model, tokenizer = load_lora_text_model(text_model_dir, device=device)
        text_model.eval()

        # 3. Load Multimodal Late Fusion Meta-Model
        fusion_model = load_fusion_model(fusion_model_path)

        return cls(
            structured_model=structured_model,
            preprocessor=preprocessor,
            text_model=text_model,
            tokenizer=tokenizer,
            fusion_model=fusion_model,
            categorical_mappings=CATEGORICAL_MAPPINGS,
            device=device,
        )

    def _prepare_single_record(self, raw: Union[EmployeeInput, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate, compute derived variables, and map categorical strings to numeric indices."""
        data = raw.to_dict() if isinstance(raw, EmployeeInput) else dict(raw)

        # 1. Continuous feature extraction and safe bounds
        tenure_months = float(data.get("tenure_months", 24.0))
        salary = float(data.get("salary", 75000.0))
        performance_score = float(np.clip(data.get("performance_score", 0.75), 0.0, 1.0))
        satisfaction_score = float(np.clip(data.get("satisfaction_score", 0.70), 0.0, 1.0))
        workload_score = float(np.clip(data.get("workload_score", 0.60), 0.0, 1.0))
        team_sentiment = float(np.clip(data.get("team_sentiment", 0.70), 0.0, 1.0))
        project_completion_rate = float(np.clip(data.get("project_completion_rate", 0.85), 0.0, 1.0))
        overtime_hours = float(max(0.0, data.get("overtime_hours", 5.0)))
        training_participation = float(np.clip(data.get("training_participation", 0.60), 0.0, 1.0))
        collaboration_score = float(np.clip(data.get("collaboration_score", 0.75), 0.0, 1.0))
        email_sentiment = float(np.clip(data.get("email_sentiment", 0.70), 0.0, 1.0))
        slack_activity = float(np.clip(data.get("slack_activity", 0.65), 0.0, 1.0))
        meeting_participation = float(np.clip(data.get("meeting_participation", 0.70), 0.0, 1.0))
        goal_achievement_rate = float(np.clip(data.get("goal_achievement_rate", 0.80), 0.0, 1.0))
        stress_level = float(np.clip(data.get("stress_level", 0.50), 0.0, 1.0))
        role_complexity_score = float(np.clip(data.get("role_complexity_score", 0.65), 0.0, 1.0))
        career_progression_score = float(np.clip(data.get("career_progression_score", 0.60), 0.0, 1.0))

        # 2. Derived features
        tech_skills = data.get("technical_skills")
        if data.get("num_technical_skills") is not None:
            num_technical_skills = float(data["num_technical_skills"])
        elif isinstance(tech_skills, (list, tuple)):
            num_technical_skills = float(len(tech_skills))
        else:
            num_technical_skills = 3.0

        soft_skills = data.get("soft_skills")
        if data.get("num_soft_skills") is not None:
            num_soft_skills = float(data["num_soft_skills"])
        elif isinstance(soft_skills, (list, tuple)):
            num_soft_skills = float(len(soft_skills))
        else:
            num_soft_skills = 2.0

        tenure_years = tenure_months / 12.0
        workload_stress_interaction = workload_score * stress_level
        satisfaction_workload_gap = satisfaction_score - workload_score
        overtime_intensity = overtime_hours / (overtime_hours + 40.0)
        engagement_score = (slack_activity + meeting_participation + collaboration_score) / 3.0

        # 3. Categorical string-to-index mapping with safe defaults
        def _map_cat(col_name: str, raw_val: Any) -> int:
            if f"{col_name}_idx" in data and data[f"{col_name}_idx"] is not None:
                return int(data[f"{col_name}_idx"])
            mapping = self.categorical_mappings.get(col_name, {})
            str_val = str(raw_val).strip() if raw_val is not None else ""
            return mapping.get(str_val, 0)

        dept_idx = _map_cat("department", data.get("department"))
        job_level_idx = _map_cat("job_level", data.get("job_level"))
        role_idx = _map_cat("role", data.get("role"))
        comm_idx = _map_cat("communication_patterns", data.get("communication_patterns"))
        persona_idx = _map_cat("persona_name", data.get("persona_name"))

        return {
            "tenure_months": np.array([tenure_months], dtype=np.float32),
            "salary": np.array([salary], dtype=np.float32),
            "performance_score": np.array([performance_score], dtype=np.float32),
            "satisfaction_score": np.array([satisfaction_score], dtype=np.float32),
            "workload_score": np.array([workload_score], dtype=np.float32),
            "team_sentiment": np.array([team_sentiment], dtype=np.float32),
            "project_completion_rate": np.array([project_completion_rate], dtype=np.float32),
            "overtime_hours": np.array([overtime_hours], dtype=np.float32),
            "training_participation": np.array([training_participation], dtype=np.float32),
            "collaboration_score": np.array([collaboration_score], dtype=np.float32),
            "email_sentiment": np.array([email_sentiment], dtype=np.float32),
            "slack_activity": np.array([slack_activity], dtype=np.float32),
            "meeting_participation": np.array([meeting_participation], dtype=np.float32),
            "goal_achievement_rate": np.array([goal_achievement_rate], dtype=np.float32),
            "stress_level": np.array([stress_level], dtype=np.float32),
            "role_complexity_score": np.array([role_complexity_score], dtype=np.float32),
            "career_progression_score": np.array([career_progression_score], dtype=np.float32),
            "num_technical_skills": np.array([num_technical_skills], dtype=np.float32),
            "num_soft_skills": np.array([num_soft_skills], dtype=np.float32),
            "tenure_years": np.array([tenure_years], dtype=np.float32),
            "workload_stress_interaction": np.array([workload_stress_interaction], dtype=np.float32),
            "satisfaction_workload_gap": np.array([satisfaction_workload_gap], dtype=np.float32),
            "overtime_intensity": np.array([overtime_intensity], dtype=np.float32),
            "engagement_score": np.array([engagement_score], dtype=np.float32),
            "department_idx": np.array([dept_idx], dtype=int),
            "job_level_idx": np.array([job_level_idx], dtype=int),
            "role_idx": np.array([role_idx], dtype=int),
            "communication_patterns_idx": np.array([comm_idx], dtype=int),
            "persona_name_idx": np.array([persona_idx], dtype=int),
        }

    def predict_single(
        self,
        employee_input: Union[EmployeeInput, Dict[str, Any]],
        threshold_override: Optional[float] = None,
    ) -> RiskPredictionResult:
        """Predict multimodal attrition risk for a single employee record."""
        data = employee_input.to_dict() if isinstance(employee_input, EmployeeInput) else dict(employee_input)
        emp_id = data.get("employee_id", "EMP-00000")
        feedback_text = str(data.get("recent_feedback", "") or "").strip()
        if not feedback_text:
            feedback_text = "No written feedback provided."

        # 1. Structured Tabular Forward Pass
        feat_dict = self._prepare_single_record(data)
        X_mat = self.preprocessor.transform(feat_dict)
        X_tensor = torch.tensor(X_mat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits_s = self.structured_model(X_tensor)
            p_struct = float(torch.sigmoid(logits_s).cpu().numpy().item())

        # 2. Text NLP Forward Pass
        encoded = self.tokenizer(
            [feedback_text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits_t = out.logits
            p_text = float(torch.sigmoid(logits_t).cpu().numpy().item())

        # 3. Multimodal Late Fusion
        p_fusion = float(self.fusion_model.predict_proba(np.array([p_struct]), np.array([p_text]))[0])

        # 4. Decision and Tier Classification
        threshold = threshold_override if threshold_override is not None else self.fusion_model.optimal_threshold
        risk_decision = int(p_fusion >= threshold)
        risk_tier = RiskTier.from_probability(p_fusion, threshold=threshold).value

        # 5. Metadata and Attribution Breakdown
        coefs = self.fusion_model.get_coefficients()
        logit_s = float(safe_logit(p_struct))
        logit_t = float(safe_logit(p_text))
        b0 = coefs.get("intercept_b0", -0.9703)
        w_s = coefs.get("w_structured", 0.1810)
        w_t = coefs.get("w_text", 0.1470)

        breakdown = {
            "structured_weight": w_s,
            "text_weight": w_t,
            "intercept": b0,
            "structured_logit": round(logit_s, 4),
            "text_logit": round(logit_t, 4),
            "structured_contribution": round(w_s * logit_s, 4),
            "text_contribution": round(w_t * logit_t, 4),
        }

        summary = (
            f"Employee {emp_id} classified as {risk_tier} RISK "
            f"(Fused Risk Probability: {p_fusion:.2%}, Decision Threshold: {threshold:.2f}). "
            f"Structured Signal: {p_struct:.2%}, Text Burnout Signal: {p_text:.2%}."
        )

        return RiskPredictionResult(
            employee_id=emp_id,
            fused_risk_probability=round(p_fusion, 4),
            structured_risk_probability=round(p_struct, 4),
            text_risk_probability=round(p_text, 4),
            risk_prediction=risk_decision,
            risk_tier=risk_tier,
            decision_threshold=round(threshold, 4),
            modality_breakdown=breakdown,
            summary=summary,
        )

    def predict_batch(
        self,
        records: Union[List[EmployeeInput], List[Dict[str, Any]], pd.DataFrame],
        threshold_override: Optional[float] = None,
        batch_size: int = 64,
    ) -> List[RiskPredictionResult]:
        """Predict multimodal attrition risk for a batch of employee records."""
        if isinstance(records, pd.DataFrame):
            raw_list = records.to_dict(orient="records")
        else:
            raw_list = [r.to_dict() if isinstance(r, EmployeeInput) else dict(r) for r in records]

        if not raw_list:
            return []

        results = []
        for item in raw_list:
            res = self.predict_single(item, threshold_override=threshold_override)
            results.append(res)

        return results
