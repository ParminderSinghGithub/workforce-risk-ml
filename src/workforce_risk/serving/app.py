"""FastAPI application for Sentinel — Multimodal Workforce Risk Intelligence Platform."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from workforce_risk.inference.predictor import WorkforceRiskPredictor
from workforce_risk.inference.schemas import EmployeeInput
from workforce_risk.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EmployeePredictionRequest,
    HealthResponse,
    ModalityBreakdown,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager loading model predictor once at server startup."""
    print("[Sentinel Startup] Initializing WorkforceRiskPredictor from saved disk artifacts...")
    try:
        predictor = WorkforceRiskPredictor.from_artifacts(device_str="cpu")
        app.state.predictor = predictor
        print("[Sentinel Startup] WorkforceRiskPredictor loaded successfully on CPU (Offline mode ready).")
    except Exception as e:
        print(f"[Sentinel Startup Error] Failed to load predictor artifacts: {e}")
        app.state.predictor = None

    yield

    print("[Sentinel Shutdown] Releasing model resources.")
    app.state.predictor = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Sentinel — Multimodal Workforce Risk Intelligence Platform API",
        description="Production inference and serving API for enterprise multimodal employee attrition and burnout risk prediction.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for local and web frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["General"])
    async def root() -> Dict[str, str]:
        """Root endpoint returning service identity and documentation pointers."""
        return {
            "service": "Sentinel — Multimodal Workforce Risk Intelligence Platform",
            "version": "0.1.0",
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(request: Request) -> HealthResponse:
        """Health check endpoint confirming that required artifacts are loaded and ready for offline inference."""
        predictor: WorkforceRiskPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model predictor artifacts are not loaded or unavailable.",
            )

        models_status = {
            "structured_mlp": predictor.structured_model is not None,
            "text_distilbert_lora": predictor.text_model is not None,
            "multimodal_late_fusion": predictor.fusion_model is not None and predictor.fusion_model.is_fitted,
        }

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            device=str(predictor.device),
            models_loaded=models_status,
            decision_threshold=predictor.fusion_model.optimal_threshold,
            offline_mode=True,
        )

    @app.get("/model-info", tags=["Metadata"])
    @app.get("/api/v1/model-info", tags=["Metadata"])
    async def get_model_info() -> Dict[str, Any]:
        """Expose technical architecture metadata, fusion coefficients, and holdout benchmarks."""
        eval_path = Path("artifacts/fusion/evaluation_summary.json").resolve()
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        else:
            eval_data = {}

        return {
            "platform": "Sentinel — Multimodal Workforce Risk Intelligence Platform",
            "version": "0.1.0",
            "architecture": {
                "structured_branch": "PyTorch StructuredMLP (Embedding + BatchNorm + Dense + Dropout)",
                "text_branch": "DistilBERT-base-uncased + PEFT/LoRA (Sequence Classifier, r=16, alpha=32)",
                "fusion_mechanism": "Calibrated Logistic Meta-Regression over unimodal log-odds",
            },
            "evaluation_benchmarks": eval_data,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict_single(
        payload: EmployeePredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        """Evaluate multimodal workforce attrition risk for a single employee record."""
        predictor: WorkforceRiskPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model predictor is not initialized.",
            )

        try:
            raw_dict = payload.model_dump()
            override = payload.threshold_override
            res = predictor.predict_single(raw_dict, threshold_override=override)

            return PredictionResponse(
                employee_id=res.employee_id,
                fused_risk_probability=res.fused_risk_probability,
                structured_risk_probability=res.structured_risk_probability,
                text_risk_probability=res.text_risk_probability,
                risk_prediction=res.risk_prediction,
                risk_tier=res.risk_tier,
                decision_threshold=res.decision_threshold,
                modality_breakdown=ModalityBreakdown(**res.modality_breakdown),
                summary=res.summary,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Inference error processing employee record: {str(e)}",
            )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    async def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ) -> BatchPredictionResponse:
        """Evaluate multimodal workforce attrition risk for a batch of employee records."""
        predictor: WorkforceRiskPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model predictor is not initialized.",
            )

        try:
            records = [emp.model_dump() for emp in payload.employees]
            batch_override = payload.threshold_override
            results = predictor.predict_batch(records, threshold_override=batch_override)

            formatted = [
                PredictionResponse(
                    employee_id=r.employee_id,
                    fused_risk_probability=r.fused_risk_probability,
                    structured_risk_probability=r.structured_risk_probability,
                    text_risk_probability=r.text_risk_probability,
                    risk_prediction=r.risk_prediction,
                    risk_tier=r.risk_tier,
                    decision_threshold=r.decision_threshold,
                    modality_breakdown=ModalityBreakdown(**r.modality_breakdown),
                    summary=r.summary,
                )
                for r in results
            ]

            return BatchPredictionResponse(
                total_predictions=len(formatted),
                predictions=formatted,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Inference error during batch processing: {str(e)}",
            )

    # Mount compiled frontend distribution if available for full-stack deployment
    dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app


app = create_app()

