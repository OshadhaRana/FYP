import React, { useState } from "react";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const sampleTx = {
    step: 10,
    type: "PAYMENT",
    amount: 1200.0,
    oldbalanceOrg: 5000.0,
    newbalanceOrig: 3800.0,
    oldbalanceDest: 1200.0,
    newbalanceDest: 2400.0
  };

  const callAPI = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sampleTx)
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (probability) => {
    if (probability > 0.5) return "#ef4444";
    if (probability > 0.2) return "#f59e0b";
    return "#10b981";
  };

  const getRiskLevel = (probability) => {
    if (probability > 0.5) return "High Risk";
    if (probability > 0.2) return "Medium Risk";
    return "Low Risk";
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: 24, maxWidth: 800, margin: "0 auto" }}>
      <h1>XAI-FinCrime MVP — Multi-Model Ensemble</h1>
      <p>Fraud detection using Random Forest, XGBoost, LSTM, and CNN models.</p>

      <div style={{ background: "#f3f4f6", padding: 16, borderRadius: 8, marginBottom: 24 }}>
        <h3>Sample Transaction</h3>
        <pre style={{ background: "white", padding: 12, borderRadius: 4, fontSize: 14 }}>
          {JSON.stringify(sampleTx, null, 2)}
        </pre>
      </div>

      <button
        onClick={callAPI}
        disabled={loading}
        style={{
          background: "#3b82f6",
          color: "white",
          padding: "12px 24px",
          border: "none",
          borderRadius: 8,
          fontSize: 16,
          cursor: loading ? "not-allowed" : "pointer",
          opacity: loading ? 0.6 : 1
        }}
      >
        {loading ? "Analyzing..." : "Predict Fraud Probability"}
      </button>

      {result && (
        <div style={{ marginTop: 32 }}>
          <div
            style={{
              background: getRiskColor(result.fraud_probability),
              color: "white",
              padding: 24,
              borderRadius: 8,
              marginBottom: 24
            }}
          >
            <h2 style={{ margin: 0 }}>
              {getRiskLevel(result.fraud_probability)}
            </h2>
            <p style={{ fontSize: 32, fontWeight: "bold", margin: "8px 0" }}>
              {(result.fraud_probability * 100).toFixed(2)}%
            </p>
            <p style={{ margin: 0, opacity: 0.9 }}>
              Fraud Probability
            </p>
          </div>

          {result.model_predictions && (
            <div style={{ background: "#f3f4f6", padding: 24, borderRadius: 8, marginBottom: 24 }}>
              <h3 style={{ marginTop: 0 }}>Individual Model Predictions</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {Object.entries(result.model_predictions).map(([model, prob]) => (
                  <div
                    key={model}
                    style={{
                      background: "white",
                      padding: 16,
                      borderRadius: 8,
                      border: `2px solid ${getRiskColor(prob)}`
                    }}
                  >
                    <div style={{ fontSize: 14, fontWeight: "bold", textTransform: "uppercase", marginBottom: 8 }}>
                      {model.replace("_", " ")}
                    </div>
                    <div style={{ fontSize: 24, fontWeight: "bold", color: getRiskColor(prob) }}>
                      {(prob * 100).toFixed(2)}%
                    </div>
                    {result.ensemble_weights && (
                      <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
                        Weight: {(result.ensemble_weights[model] * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.ensemble_weights && (
            <div style={{ background: "#f3f4f6", padding: 24, borderRadius: 8 }}>
              <h3 style={{ marginTop: 0 }}>Ensemble Configuration</h3>
              <p style={{ color: "#6b7280", marginBottom: 16 }}>
                The final prediction is a weighted average of all model predictions:
              </p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {Object.entries(result.ensemble_weights).map(([model, weight]) => (
                  <div
                    key={model}
                    style={{
                      background: "white",
                      padding: "8px 16px",
                      borderRadius: 4,
                      fontSize: 14
                    }}
                  >
                    <span style={{ fontWeight: "bold", textTransform: "uppercase" }}>
                      {model}:
                    </span>{" "}
                    {(weight * 100).toFixed(0)}%
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.note && (
            <div style={{ marginTop: 16, padding: 16, background: "#fef3c7", borderRadius: 8, color: "#92400e" }}>
              ⚠️ {result.note}
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: 48, padding: 24, background: "#f9fafb", borderRadius: 8, borderLeft: "4px solid #3b82f6" }}>
        <h3 style={{ marginTop: 0 }}>About the Models</h3>
        <ul style={{ color: "#6b7280", lineHeight: 1.8 }}>
          <li><strong>Random Forest</strong>: Ensemble of decision trees for robust predictions</li>
          <li><strong>XGBoost</strong>: Gradient boosting for high accuracy</li>
          <li><strong>LSTM</strong>: Recurrent neural network for sequential pattern detection</li>
          <li><strong>CNN</strong>: Convolutional neural network for feature extraction</li>
        </ul>
        <p style={{ color: "#6b7280", marginBottom: 0 }}>
          The multi-model ensemble combines the strengths of all models for superior fraud detection accuracy.
        </p>
      </div>
    </div>
  );
}
