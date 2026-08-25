import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, Shield, Database, FileText, Layers, Award } from 'lucide-react';
import { fetchModelInfo } from '../services/api';

export const MethodologyView: React.FC = () => {
  const [, setModelInfo] = useState<any>(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch((err) => console.log('Model info fallback to embedded:', err));
  }, []);

  return (
    <div>
      <div className="page-header">
        <div className="page-title">
          <Cpu size={26} color="var(--brand-accent)" />
          <span>Technical Architecture & ML Methodology</span>
        </div>
        <div className="page-description">
          Sentinel's multimodal modeling pipeline combines deep tabular embeddings, parameter-efficient transformer NLP, and calibrated late-fusion meta-regression.
        </div>
      </div>

      {/* Model Benchmark Table */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Award size={18} color="var(--risk-elevated)" />
              Untouched Holdout Test Partition Benchmark Comparison ($N = 8,463$)
            </div>
            <div className="card-subtitle">
              Strict dual-holdout test evaluation (0% employee overlap, 0% template overlap with training)
            </div>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Operating Threshold (&tau;*)</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>Log-Loss</th>
                <th>Brier Score</th>
                <th>Calibration Benefit</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>1. Structured Tabular MLP</strong></td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.42</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.5106</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.2985</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.6584</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.2327</td>
                <td>Tabular Baseline</td>
              </tr>
              <tr>
                <td><strong>2. TF-IDF Text Baseline</strong></td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.18</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.5199</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.3007</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>1.1337</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.3871</td>
                <td>Uncalibrated N-Gram</td>
              </tr>
              <tr>
                <td><strong>3. DistilBERT + PEFT/LoRA</strong></td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.15</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.5452</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.3189</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>1.0089</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>0.3554</td>
                <td>High NLP Sensitivity</td>
              </tr>
              <tr style={{ backgroundColor: 'rgba(37, 99, 235, 0.12)', borderLeft: '4px solid #2563eb' }}>
                <td><strong style={{ color: '#60a5fa' }}>4. Sentinel Multimodal Late Fusion</strong></td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>0.22</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#60a5fa' }}>0.5452</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#60a5fa' }}>0.3196</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#10b981' }}>0.5983</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#10b981' }}>0.2042</td>
                <td style={{ color: '#10b981', fontWeight: 600 }}>Optimal Error Floor</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 3 Pillars of Architecture */}
      <div className="grid-3" style={{ marginBottom: '2rem' }}>
        {/* Tabular Branch */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Database size={18} color="#3b82f6" />
              Tabular Feature Pipeline
            </div>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <p><strong>Model:</strong> PyTorch Multi-Layer Perceptron (MLP)</p>
            <p style={{ marginTop: '0.5rem' }}><strong>Features:</strong> 17 Continuous Telemetry Features + 7 Derived Interaction Terms + 5 Frequency-Indexed Categorical Dimensions.</p>
            <p style={{ marginTop: '0.5rem' }}><strong>Standardization:</strong> Feature standardizer fitted strictly on training partition without out-of-fold data leakage.</p>
          </div>
        </div>

        {/* NLP Transformer Branch */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <FileText size={18} color="#f97316" />
              PEFT / LoRA Transformer
            </div>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <p><strong>Base Backbone:</strong> DistilBERT-base-uncased (66M parameters)</p>
            <p style={{ marginTop: '0.5rem' }}><strong>Adapter Architecture:</strong> Low-Rank Adaptation (LoRA, r=16, &alpha;=32) trained on 679k feedback reviews for 3 epochs.</p>
            <p style={{ marginTop: '0.5rem' }}><strong>Target Task:</strong> High burnout classification (P(burnout_risk &ge; 0.75 | feedback)).</p>
          </div>
        </div>

        {/* Late Fusion Head */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Layers size={18} color="#2563eb" />
              Calibrated Late Fusion
            </div>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <p><strong>Meta-Regressor:</strong> Logistic Regression over unimodal log-odds</p>
            <p style={{ marginTop: '0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', backgroundColor: 'var(--bg-input)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
              logit(p_fused) = -0.9703 + 0.1810·logit(p_struct) + 0.1470·logit(p_text)
            </p>
            <p style={{ marginTop: '0.5rem' }}><strong>Thresholding:</strong> Optimal operating cutoff (&tau;* = 0.2189) tuned on validation set to maximize enterprise risk recall.</p>
          </div>
        </div>
      </div>

      {/* Enterprise Guarantees */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Shield size={18} color="var(--risk-low)" />
            Data Leakage Controls & Rigor Protocol
          </div>
        </div>
        <div className="grid-2" style={{ gap: '1rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <CheckCircle2 size={16} color="var(--risk-low)" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
            <div>
              <strong>Strict Target Exclusion:</strong> Columns `turnover_reason`, `turnover_probability_generated`, and `risk_factors_summary` are permanently excluded from feature ingestion.
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <CheckCircle2 size={16} color="var(--risk-low)" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
            <div>
              <strong>Disjoint Review Templates:</strong> Text splits are grouped by feedback template to ensure 0% lexical template memorization across train, val, and test.
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <CheckCircle2 size={16} color="var(--risk-low)" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
            <div>
              <strong>Offline Artifact Serving:</strong> All models execute offline from local safetensors, PyTorch checkpoints, and joblib binaries without runtime Hugging Face queries.
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
            <CheckCircle2 size={16} color="var(--risk-low)" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
            <div>
              <strong>Dual-Holdout Validation Integrity:</strong> Threshold optimization and fusion fitting are computed exclusively on validation partitions; test set remains untouched.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
