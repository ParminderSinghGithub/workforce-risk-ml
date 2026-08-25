import {
  BatchPredictionRequest,
  BatchPredictionResponse,
  EmployeePredictionRequest,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE}/model-info`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Model info fetch failed: ${response.statusText}`);
  }
  return response.json();
}

export async function predictEmployee(
  payload: EmployeePredictionRequest
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(
      response.status,
      typeof errorData.detail === 'string'
        ? errorData.detail
        : `Prediction failed with HTTP ${response.status}`
    );
  }
  return response.json();
}

export async function predictBatch(
  employees: EmployeePredictionRequest[],
  thresholdOverride?: number | null
): Promise<BatchPredictionResponse> {
  const payload: BatchPredictionRequest = {
    employees,
    threshold_override: thresholdOverride,
  };

  const response = await fetch(`${API_BASE}/predict/batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(
      response.status,
      typeof errorData.detail === 'string'
        ? errorData.detail
        : `Batch prediction failed with HTTP ${response.status}`
    );
  }
  return response.json();
}
