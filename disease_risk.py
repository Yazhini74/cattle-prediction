"""Standalone rule-based disease risk analytics for the Streamlit dashboard.

This module deliberately has no dependency on the production prediction models or
their pickle files.  Its scores are management-support indicators, not diagnoses.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import plotly.graph_objects as go
import streamlit as st


RISK_COLORS = {
    "Low Risk": "#10B981",
    "Moderate Risk": "#F59E0B",
    "High Risk": "#EF4444",
}


def _bounded(value: float, lower: float, upper: float) -> float:
    """Convert a value to a 0-1 range while safely handling boundary values."""
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _risk_level(probability: float) -> str:
    if probability >= 65:
        return "High Risk"
    if probability >= 35:
        return "Moderate Risk"
    return "Low Risk"


def predict_disease_risks(
    ptascs: float,
    productive_life: float,
    daughter_pregnancy_rate: float,
    age_months: int,
    lactation_number: int,
    body_condition_score: float,
    temperature_stress_index: int,
) -> Dict[str, Dict[str, Union[str, float]]]:
    """Return transparent, rule-based disease-risk estimates.

    PTA Somatic Cell Score, Productive Life, and Daughter Pregnancy Rate are
    existing genomic inputs.  The remaining inputs are current-animal management
    factors.  Scores are deliberately bounded to 5-95% and are intended to guide
    monitoring priorities rather than replace veterinary diagnosis.
    """
    scs_risk = _bounded(ptascs, 2.6, 4.1)
    low_longevity = 1.0 - _bounded(productive_life, -3.0, 7.0)
    poor_fertility = 1.0 - _bounded(daughter_pregnancy_rate, -4.0, 4.0)
    age_risk = _bounded(age_months, 24.0, 120.0)
    lactation_risk = _bounded(lactation_number, 1.0, 7.0)
    thin_condition = 1.0 - _bounded(body_condition_score, 2.0, 3.0)
    over_condition = _bounded(body_condition_score, 3.25, 4.5)
    heat_stress = _bounded(temperature_stress_index, 68.0, 90.0)

    probabilities = {
        "Mastitis": 100 * (0.46 * scs_risk + 0.18 * low_longevity + 0.16 * age_risk
                            + 0.10 * lactation_risk + 0.10 * heat_stress),
        "Ketosis": 100 * (0.33 * thin_condition + 0.22 * over_condition + 0.20 * lactation_risk
                           + 0.15 * heat_stress + 0.10 * low_longevity),
        "Milk Fever": 100 * (0.34 * age_risk + 0.28 * lactation_risk + 0.25 * over_condition
                              + 0.13 * heat_stress),
        "Reproductive Disorders": 100 * (0.42 * poor_fertility + 0.20 * heat_stress
                                            + 0.16 * over_condition + 0.12 * thin_condition
                                            + 0.10 * age_risk),
    }
    recommendations = {
        "Mastitis": {
            "High Risk": "Increase udder health monitoring, review milking hygiene, and investigate elevated cell-count trends.",
            "Moderate Risk": "Maintain consistent udder preparation and review somatic cell counts at routine checks.",
            "Low Risk": "Continue current udder-health and milking sanitation protocols.",
        },
        "Ketosis": {
            "High Risk": "Review energy balance and feeding strategy, especially through the transition period.",
            "Moderate Risk": "Monitor dry-matter intake and body-condition changes around calving.",
            "Low Risk": "Maintain the current transition diet and routine metabolic monitoring.",
        },
        "Milk Fever": {
            "High Risk": "Review close-up ration mineral balance and plan calcium support with the herd veterinarian.",
            "Moderate Risk": "Monitor fresh-cow calcium status and confirm the transition ration is appropriate.",
            "Low Risk": "Continue standard transition-cow mineral management.",
        },
        "Reproductive Disorders": {
            "High Risk": "Prioritize heat-stress mitigation and reproductive examinations; review breeding and transition management.",
            "Moderate Risk": "Track breeding outcomes and minimize heat stress during the breeding window.",
            "Low Risk": "Maintain routine reproductive monitoring and breeding records.",
        },
    }

    results: Dict[str, Dict[str, Union[str, float]]] = {}
    for disease, raw_probability in probabilities.items():
        probability = float(np.clip(raw_probability, 5.0, 95.0))
        level = _risk_level(probability)
        results[disease] = {
            "probability": round(probability, 1),
            "risk_level": level,
            "recommendation": recommendations[disease][level],
        }
    return results


def _risk_gauge(disease: str, probability: float, level: str) -> go.Figure:
    color = RISK_COLORS[level]
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 28}},
        title={"text": disease, "font": {"color": "#E2E8F0", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8", "tickfont": {"color": "#94A3B8", "size": 9}},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": "rgba(255,255,255,0.06)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(16,185,129,0.10)"},
                {"range": [35, 65], "color": "rgba(245,158,11,0.10)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.10)"},
            ],
        },
    ))
    figure.update_layout(
        height=190,
        margin=dict(l=12, r=12, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
    )
    return figure


def render_disease_risk_module(ptascs: float, productive_life: float, daughter_pregnancy_rate: float) -> None:
    """Render the opt-in disease analytics controls and results in Streamlit."""
    st.markdown("<div class='panel-subheader'>Disease Risk Prediction</div>", unsafe_allow_html=True)
    st.caption("Rule-based decision support using existing genomic traits and current-animal management factors. This is not a veterinary diagnosis.")

    input_cols = st.columns(4)
    with input_cols[0]:
        age_months = st.number_input("Age of Animal (months)", min_value=18, max_value=180, value=48, step=1, key="disease_age")
    with input_cols[1]:
        lactation_number = st.number_input("Lactation Number", min_value=1, max_value=12, value=2, step=1, key="disease_lactation")
    with input_cols[2]:
        body_condition_score = st.slider("Body Condition Score", 1.0, 5.0, value=3.0, step=0.1, key="disease_bcs")
    with input_cols[3]:
        temperature_stress_index = st.slider("Temperature Stress Index", 50, 100, value=68, step=1, key="disease_tsi")

    st.caption(
        f"Reused genomic inputs: Somatic Cell Score {ptascs:.2f} · Productive Life {productive_life:.1f} · Daughter Pregnancy Rate {daughter_pregnancy_rate:.1f}"
    )
    if st.button("Calculate Disease Risks", use_container_width=True, key="calculate_disease_risks"):
        st.session_state["show_disease_risks"] = True

    if not st.session_state.get("show_disease_risks", False):
        return

    results = predict_disease_risks(
        ptascs, productive_life, daughter_pregnancy_rate, age_months,
        lactation_number, body_condition_score, temperature_stress_index,
    )
    risk_cols = st.columns(4)
    for column, (disease, result) in zip(risk_cols, results.items()):
        probability = float(result["probability"])
        level = str(result["risk_level"])
        color = RISK_COLORS[level]
        with column:
            st.plotly_chart(_risk_gauge(disease, probability, level), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div class='glass-card' style='min-height: 115px; padding: 14px !important;'>"
                f"<div class='kpi-title'>{level}</div>"
                f"<div style='font-size: 0.78rem; color: #CBD5E1; line-height: 1.45;'>{result['recommendation']}</div>"
                f"<div style='width: 100%; background: rgba(255,255,255,0.08); height: 6px; border-radius: 3px; margin-top: 12px; overflow: hidden;'>"
                f"<div style='width: {probability:.1f}%; background: {color}; height: 100%; border-radius: 3px;'></div></div>"
                f"<div style='font-size: 0.72rem; color: {color}; margin-top: 5px;'>{probability:.1f}% estimated risk</div></div>",
                unsafe_allow_html=True,
            )
