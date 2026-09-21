
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CRICKCAST | Score & Win Probability",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM UI
# ============================================================
st.markdown(
    """
<style>
:root {
    --bg: #070b14;
    --panel: #0d1422;
    --panel2: #111a2b;
    --line: rgba(255,255,255,.09);
    --text: #f5f7fb;
    --muted: #94a3b8;
    --cyan: #22d3ee;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #fb7185;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(34,211,238,.08), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(52,211,153,.07), transparent 24%),
        var(--bg);
}

[data-testid="stHeader"] {
    background: rgba(7,11,20,.88);
    backdrop-filter: blur(16px);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1b 0%, #0d1422 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: #e9eef7;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 28px 30px;
    background:
        linear-gradient(135deg, rgba(34,211,238,.08), rgba(52,211,153,.02) 55%, rgba(15,23,42,.4)),
        rgba(13,20,34,.88);
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
    margin-bottom: 18px;
}

.hero-kicker {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: .16em;
    font-weight: 800;
    font-size: .72rem;
    margin-bottom: 8px;
}

.hero-title {
    color: var(--text);
    font-size: clamp(2rem, 4vw, 3.45rem);
    font-weight: 850;
    line-height: 1.0;
    letter-spacing: -.04em;
    margin: 0;
}

.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 12px;
    max-width: 880px;
    line-height: 1.6;
}

.status-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: .76rem;
    font-weight: 750;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.035);
    color: #dbe5f3;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    background: var(--green);
    box-shadow: 0 0 12px rgba(52,211,153,.7);
}

.section-title {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 18px 0 9px 2px;
}

.section-sub {
    color: var(--muted);
    font-size: .84rem;
    margin: -4px 0 12px 2px;
}

.metric-card {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 17px 18px 15px;
    background: linear-gradient(180deg, rgba(17,26,43,.95), rgba(13,20,34,.95));
    min-height: 116px;
    box-shadow: 0 10px 30px rgba(0,0,0,.14);
}

.metric-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
    font-size: .68rem;
    font-weight: 800;
}

.metric-value {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 850;
    letter-spacing: -.035em;
    margin-top: 8px;
}

.metric-note {
    color: #71829a;
    font-size: .74rem;
    margin-top: 5px;
}

.metric-accent {
    position: absolute;
    top: 0;
    left: 0;
    width: 5px;
    height: 100%;
    border-radius: 18px 0 0 18px;
    background: linear-gradient(180deg, var(--cyan), var(--green));
}

.prob-card {
    border: 1px solid rgba(34,211,238,.18);
    border-radius: 18px;
    padding: 20px;
    background: radial-gradient(circle at top right, rgba(34,211,238,.10), transparent 38%), #0d1422;
}

.big-prob {
    font-size: 3.2rem;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -.06em;
    color: white;
}

.big-prob-sub {
    color: var(--muted);
    font-size: .8rem;
    margin-top: 8px;
}

.pill {
    display: inline-block;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 4px 9px;
    color: #dbeafe;
    font-size: .68rem;
    font-weight: 800;
    background: rgba(255,255,255,.035);
}

.footer-note {
    border-top: 1px solid var(--line);
    margin-top: 40px;
    padding-top: 16px;
    color: #718096;
    font-size: .75rem;
    line-height: 1.6;
}

div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--line);
}

button[data-baseweb="tab"] {
    color: #8ea0b8;
    font-weight: 750;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(34,211,238,.25);
    background: linear-gradient(135deg, rgba(34,211,238,.15), rgba(52,211,153,.08));
    color: #ecfeff;
    font-weight: 800;
    min-height: 44px;
}

.stButton > button:hover {
    border-color: rgba(34,211,238,.5);
    background: linear-gradient(135deg, rgba(34,211,238,.22), rgba(52,211,153,.12));
}

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(13,20,34,.72);
}

.small-muted {
    color: #718096;
    font-size: .74rem;
}

@media (max-width: 900px) {
    .hero { padding: 22px; }
    .metric-value { font-size: 1.55rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA + MODEL
# ============================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_cricket_data(num_matches=200):
    rng = np.random.default_rng(42)
    data = []

    for match_id in range(1, num_matches + 1):
        first_inn_score = int(rng.normal(120, 15))
        target = first_inn_score + 1

        current_runs = 0
        current_wickets = 0
        runs_per_over_history = []
        wickets_per_over_history = []

        chase_strength = rng.choice([1.0, 1.2, 1.4])

        for over in range(1, 21):
            if current_wickets >= 10:
                break

            base_probs = [0.05, 0.15, 0.22, 0.20, 0.15, 0.10, 0.08, 0.04, 0.01]
            runs_options = [0, 1, 2, 4, 6, 8, 12, 15, 20]

            runs_this_over = int(rng.choice(runs_options, p=base_probs) * chase_strength)
            runs_this_over = int(min(36, runs_this_over))

            wickets_this_over = int(
                rng.choice([0, 1, 2, 3], p=[0.80, 0.15, 0.04, 0.01])
            )

            current_runs += runs_this_over
            current_wickets = min(10, current_wickets + wickets_this_over)

            runs_per_over_history.append(runs_this_over)
            wickets_per_over_history.append(wickets_this_over)

            crr = current_runs / over
            rrr = (target - current_runs) / (20 - over) if over < 20 else 0
            last_5_runs = sum(runs_per_over_history[-5:])
            last_5_wickets = sum(wickets_per_over_history[-5:])

            data.append(
                {
                    "match_id": match_id,
                    "over": over,
                    "runs": current_runs,
                    "wickets": current_wickets,
                    "current_run_rate": round(crr, 2),
                    "last_5_over_runs": last_5_runs,
                    "last_5_over_wickets": last_5_wickets,
                    "target": target,
                    "required_run_rate": round(max(0, rrr), 2),
                    "final_score": 0,
                    "won": 0,
                }
            )

            if current_runs >= target:
                break

        match_rows = [row for row in data if row["match_id"] == match_id]
        final_runs = current_runs
        won_status = int(final_runs >= target)

        for row in match_rows:
            row["final_score"] = final_runs
            row["won"] = won_status

    return pd.DataFrame(data)


FEATURES = [
    "over",
    "runs",
    "wickets",
    "current_run_rate",
    "last_5_over_runs",
    "last_5_over_wickets",
]


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df[FEATURES]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear"] = {"model": lr, "predictor": lambda x, m=lr: m.predict(x)}

    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train)
    X_test_poly2 = poly2.transform(X_test)
    lr_poly2 = LinearRegression()
    lr_poly2.fit(X_train_poly2, y_train)
    models["Polynomial (Degree 2)"] = {
        "model": lr_poly2,
        "transformer": poly2,
        "predictor": lambda x, m=lr_poly2, p=poly2: m.predict(p.transform(x)),
    }

    poly3 = PolynomialFeatures(degree=3)
    X_train_poly3 = poly3.fit_transform(X_train)
    X_test_poly3 = poly3.transform(X_test)
    lr_poly3 = LinearRegression()
    lr_poly3.fit(X_train_poly3, y_train)
    models["Polynomial (Degree 3)"] = {
        "model": lr_poly3,
        "transformer": poly3,
        "predictor": lambda x, m=lr_poly3, p=poly3: m.predict(p.transform(x)),
    }

    metric_rows = []
    for name, pack in models.items():
        pred = pack["predictor"](X_test)
        metric_rows.append(
            {
                "Model": name,
                "R²": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": math.sqrt(mean_squared_error(y_test, pred)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    best_model_name = metrics.sort_values("MAE").iloc[0]["Model"]

    # Residuals for empirical score-threshold probability estimates.
    best_pred_test = models[best_model_name]["predictor"](X_test)
    residual_std = float(np.std(y_test.to_numpy() - best_pred_test))

    return models, metrics, best_model_name, residual_std, X_test, y_test


def dynamic_win_probability(over, runs, wickets, target, crr, rrr):
    if runs >= target:
        return 1.0
    if over >= 20 or wickets >= 10:
        return 0.0 if runs < target else 1.0

    wickets_left = max(0, 10 - wickets)
    wicket_penalty = (10 - wickets_left) * 0.08

    if rrr > 0:
        ratio = crr / rrr
        prob = 0.5 * ratio - wicket_penalty
    else:
        prob = 1.0 - wicket_penalty

    return float(max(0.02, min(0.98, prob)))


@st.cache_data(show_spinner=False)
def probability_artifacts(df):
    temp = df.copy()
    temp["runs_in_over"] = temp.groupby("match_id")["runs"].diff().fillna(temp["runs"])

    prob_dist = temp["runs_in_over"].value_counts(normalize=True).sort_index()
    expected_value = float((prob_dist.index.to_numpy() * prob_dist.to_numpy()).sum())
    expected_square = float(
        ((prob_dist.index.to_numpy() ** 2) * prob_dist.to_numpy()).sum()
    )
    variance = float(expected_square - expected_value**2)

    df_chase = df[df["target"] > 0].copy()
    df_chase["ahead_of_rrr"] = (
        df_chase["current_run_rate"] >= df_chase["required_run_rate"]
    )

    p_win = float(df_chase["won"].mean())
    p_ahead = float(df_chase["ahead_of_rrr"].mean())
    p_ahead_given_win = float(
        df_chase[df_chase["won"] == 1]["ahead_of_rrr"].mean()
    )

    p_win_given_ahead = (
        (p_ahead_given_win * p_win) / p_ahead if p_ahead > 0 else 0.0
    )

    return (
        prob_dist,
        expected_value,
        variance,
        p_win,
        p_ahead,
        p_ahead_given_win,
        p_win_given_ahead,
    )


# ============================================================
# HELPERS
# ============================================================
def card(label, value, note="", accent=True):
    accent_html = '<div class="metric-accent"></div>' if accent else ""
    st.markdown(
        f"""
        <div class="metric-card">
            {accent_html}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_percent(x):
    return f"{x * 100:.1f}%"


# ============================================================
# LOAD
# ============================================================
df = generate_synthetic_cricket_data()
models, metrics, best_model_name, residual_std, X_test, y_test = train_models(df)
(
    prob_dist,
    expected_runs_over,
    variance_runs_over,
    p_win,
    p_ahead,
    p_ahead_given_win,
    p_win_given_ahead,
) = probability_artifacts(df)


# ============================================================
# SIDEBAR — MATCH CONTROL
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 2px 16px;">
            <div style="font-size:.7rem;letter-spacing:.15em;color:#22d3ee;font-weight:800;">LIVE MATCH CONSOLE</div>
            <div style="font-size:1.35rem;font-weight:900;color:#fff;margin-top:4px;">Match State</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    batting_team = st.text_input("Batting team", "India")
    opposition = st.text_input("Opponent", "Australia")
    venue = st.text_input("Venue", "Academic Demo Arena")

    st.divider()

    over_whole = st.slider("Overs completed", 1, 19, 15)
    balls = st.selectbox("Balls in current over", list(range(0, 6)), index=2)

    runs = st.number_input("Current runs", min_value=0, max_value=300, value=132, step=1)
    wickets = st.slider("Wickets lost", 0, 9, 4)

    target = st.number_input("Target", min_value=1, max_value=350, value=185, step=1)

    default_last5 = max(0, min(90, int(round((runs / max(1, over_whole)) * 5))))
    last5_runs = st.number_input(
        "Runs in last 5 overs", min_value=0, max_value=120, value=47, step=1
    )
    last5_wickets = st.slider("Wickets in last 5 overs", 0, 5, 1)

    st.divider()

    model_view = st.selectbox(
        "Prediction model",
        ["Auto (lowest MAE)", "Linear", "Polynomial (Degree 2)", "Polynomial (Degree 3)"],
        index=0,
    )

    st.markdown(
        """
        <div style="margin-top:14px;padding:12px;border:1px solid rgba(255,255,255,.08);
        border-radius:12px;background:rgba(255,255,255,.025);">
            <div style="font-weight:800;font-size:.8rem;">Data note</div>
            <div class="small-muted" style="margin-top:5px;">
                The current notebook trains on 200 synthetic matches generated with seed 42.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


effective_over = min(20.0, over_whole + balls / 6)
crr = runs / effective_over
overs_left = max(0.0, 20 - effective_over)
rrr = max(0.0, (target - runs) / overs_left) if overs_left > 0 else 0.0

input_row = pd.DataFrame(
    [{
        "over": effective_over,
        "runs": runs,
        "wickets": wickets,
        "current_run_rate": crr,
        "last_5_over_runs": last5_runs,
        "last_5_over_wickets": last5_wickets,
    }]
)

if model_view == "Auto (lowest MAE)":
    selected_model = best_model_name
else:
    selected_model = model_view

pred = float(models[selected_model]["predictor"](input_row)[0])
pred = max(float(runs), pred)
pred = min(350.0, pred)

# Cap prediction sensibly for chase context.
if target > runs:
    pred = max(pred, min(runs + overs_left * 1.0, target + 40))

# Empirical/normal residual approximation for score threshold probabilities.
sigma = max(residual_std, 8.0)
p_reach_target = float(1 - norm.cdf((target - pred) / sigma))

prob_180 = float(1 - norm.cdf((180 - pred) / sigma))
prob_200 = float(1 - norm.cdf((200 - pred) / sigma))

win_prob = dynamic_win_probability(
    effective_over, runs, wickets, target, crr, rrr
)

ahead_of_rrr = crr >= rrr if overs_left > 0 else runs >= target


# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">CRICKET ANALYTICS LAB · MACHINE LEARNING + PROBABILITY</div>
        <div class="hero-title">CRICKETIQ</div>
        <div class="hero-sub">
            Cricket Score Prediction & Dynamic Win Probability Estimation —
            a polished academic dashboard built directly from your notebook's
            synthetic match-generation, regression and probability logic.
        </div>
        <div class="status-row">
            <span class="status"><span class="status-dot"></span> MODEL READY</span>
            <span class="status">200 SYNTHETIC MATCHES</span>
            <span class="status">3 REGRESSION MODELS</span>
            <span class="status">BAYES + DYNAMIC WIN PROBABILITY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🏏 Match Console",
        "📈 Model Lab",
        "🎲 Probability Lab",
        "🧭 Match Explorer",
        "ℹ️ Project Notes",
    ]
)


# ============================================================
# TAB 1 — MATCH CONSOLE
# ============================================================
with tabs[0]:
    st.markdown('<div class="section-title">Current Match State</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">{batting_team} vs {opposition} · {venue}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Current Score", f"{runs}/{wickets}", f"after {effective_over:.1f} overs")
    with c2:
        card("Predicted Final", f"{pred:.0f}", f"{selected_model}")
    with c3:
        card("Expected Score", f"{pred:.1f}", f"model point estimate")
    with c4:
        card("Win Probability", fmt_percent(win_prob), "dynamic estimate")
    with c5:
        card("Target Reach", fmt_percent(p_reach_target), f"P(score ≥ {target})")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.95])

    with left:
        st.markdown('<div class="section-title">Match Signal</div>', unsafe_allow_html=True)
        signal_text = "AHEAD OF REQUIRED RATE" if ahead_of_rrr else "BEHIND REQUIRED RATE"
        signal_color = "#34d399" if ahead_of_rrr else "#fb7185"

        st.markdown(
            f"""
            <div class="prob-card">
                <div style="color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;">
                    LIVE CONTEXT
                </div>
                <div style="margin-top:14px;font-size:1.45rem;font-weight:900;color:{signal_color};">
                    {signal_text}
                </div>
                <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div><div class="small-muted">Current RR</div><div style="font-size:1.25rem;font-weight:850;color:#fff;">{crr:.2f}</div></div>
                    <div><div class="small-muted">Required RR</div><div style="font-size:1.25rem;font-weight:850;color:#fff;">{rrr:.2f}</div></div>
                    <div><div class="small-muted">Runs needed</div><div style="font-size:1.25rem;font-weight:850;color:#fff;">{max(0, target-runs)}</div></div>
                    <div><div class="small-muted">Overs left</div><div style="font-size:1.25rem;font-weight:850;color:#fff;">{overs_left:.1f}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="prob-card">
                <div style="color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;">
                    Bayesian Context
                </div>
                <div style="display:flex;justify-content:space-between;align-items:end;margin-top:10px;">
                    <div class="big-prob">{p_win_given_ahead*100:.0f}%</div>
                    <span class="pill">P(Win | Ahead of RRR)</span>
                </div>
                <div class="big-prob-sub">
                    Historical conditional estimate calculated from the notebook's synthetic sample.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("How the prediction is computed"):
            st.write(
                "The selected regression model uses the same six features defined in the notebook: "
                "over, runs, wickets, current run rate, last-5-over runs and last-5-over wickets."
            )
            st.write(
                "Win probability uses the notebook's dynamic run-rate/wicket heuristic. "
                "The Bayes panel separately reports P(Win | Ahead of Required Run Rate)."
            )

    with right:
        st.markdown('<div class="section-title">Score Projection</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {
                "Metric": ["Current Score", "Predicted Final", "Target"],
                "Runs": [runs, pred, target],
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=chart_df["Metric"],
                y=chart_df["Runs"],
                marker_color=["#22d3ee", "#34d399", "#fbbf24"],
                text=[f"{v:.0f}" for v in chart_df["Runs"]],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.0f} runs<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe5f3"),
            showlegend=False,
            yaxis=dict(gridcolor="rgba(255,255,255,.07)", title="Runs"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">P(Score ≥ 180)</div>
                    <div class="metric-value">{prob_180*100:.0f}%</div>
                    <div class="metric-note">model-based threshold estimate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cB:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">P(Score ≥ 200)</div>
                    <div class="metric-value">{prob_200*100:.0f}%</div>
                    <div class="metric-note">model-based threshold estimate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(max(win_prob, 0.0), 1.0))
        st.caption(f"Dynamic win probability: {win_prob*100:.1f}% · current RR {crr:.2f} vs required RR {rrr:.2f}")


# ============================================================
# TAB 2 — MODEL LAB
# ============================================================
with tabs[1]:
    st.markdown('<div class="section-title">Regression Model Laboratory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Same feature set and model families used in the uploaded notebook.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    for col, (_, row) in zip([m1, m2, m3], metrics.iterrows()):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{row['Model']}</div>
                    <div class="metric-value">R² {row['R²']:.3f}</div>
                    <div class="metric-note">MAE {row['MAE']:.2f} · RMSE {row['RMSE']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        metrics.style.format({"R²": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=metrics["Model"], y=metrics["MAE"]))
    fig.add_trace(go.Bar(name="RMSE", x=metrics["Model"], y=metrics["RMSE"]))
    fig.update_layout(
        barmode="group",
        height=390,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5f3"),
        yaxis=dict(gridcolor="rgba(255,255,255,.07)", title="Error"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Academic note — model validation"):
        st.write(
            "The uploaded notebook uses a random row-wise train/test split. Because multiple rows come "
            "from the same synthetic match, this can allow snapshots from one match to appear in both "
            "sets. For a production-grade study, a group-wise split by match_id would be more rigorous."
        )
        st.write(
            "This Streamlit version keeps the notebook's original split so the deployed app remains faithful "
            "to the submitted notebook."
        )


# ============================================================
# TAB 3 — PROBABILITY LAB
# ============================================================
with tabs[2]:
    st.markdown('<div class="section-title">Probability & Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Empirical distributions calculated directly from the generated snapshots.</div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        card("E[X] runs/over", f"{expected_runs_over:.2f}", "expected value")
    with p2:
        card("Var(X)", f"{variance_runs_over:.2f}", "variance of runs/over")
    with p3:
        card("P(Win)", fmt_percent(p_win), "historical prior")
    with p4:
        card("P(Win | Ahead RRR)", fmt_percent(p_win_given_ahead), "Bayesian conditional")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        pmf_df = prob_dist.rename("Probability").reset_index()
        pmf_df.columns = ["Runs in over", "Probability"]
        fig = px.bar(
            pmf_df,
            x="Runs in over",
            y="Probability",
            title="Empirical PMF · Runs Scored per Over",
        )
        fig.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe5f3"),
            title_font_size=15,
            yaxis=dict(gridcolor="rgba(255,255,255,.07)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        q = df["final_score"].quantile([0.25, 0.5, 0.75, 0.9]).rename(
            {0.25: "25th", 0.5: "50th", 0.75: "75th", 0.9: "90th"}
        )
        q_df = q.reset_index()
        q_df.columns = ["Quantile", "Final score"]

        fig = px.bar(
            q_df,
            x="Quantile",
            y="Final score",
            title="Final Score Quantiles",
        )
        fig.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe5f3"),
            title_font_size=15,
            yaxis=dict(gridcolor="rgba(255,255,255,.07)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Final Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(
        df.drop_duplicates("match_id"),
        x="final_score",
        nbins=24,
        marginal="box",
        title="Distribution of Final Scores",
    )
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5f3"),
        title_font_size=15,
        yaxis=dict(gridcolor="rgba(255,255,255,.07)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Covariance Matrix</div>', unsafe_allow_html=True)
    cov_cols = [
        "over",
        "runs",
        "wickets",
        "current_run_rate",
        "last_5_over_runs",
        "final_score",
    ]
    cov = df[cov_cols].cov()

    fig = px.imshow(
        cov,
        text_auto=".1f",
        aspect="auto",
        title="Feature Covariance",
    )
    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5f3"),
        title_font_size=15,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 — MATCH EXPLORER
# ============================================================
with tabs[3]:
    st.markdown('<div class="section-title">Synthetic Match Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Explore one of the generated matches and its notebook-style dynamic progression.</div>',
        unsafe_allow_html=True,
    )

    match_id = st.selectbox("Select match", sorted(df["match_id"].unique()), index=0)
    match_data = df[df["match_id"] == match_id].sort_values("over").copy()

    match_target = int(match_data["target"].iloc[0])
    match_won = bool(match_data["won"].iloc[0])

    probs = []
    for _, row in match_data.iterrows():
        probs.append(
            dynamic_win_probability(
                row["over"],
                row["runs"],
                row["wickets"],
                row["target"],
                row["current_run_rate"],
                row["required_run_rate"],
            )
            * 100
        )

    left, right = st.columns([1.65, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=match_data["over"],
                y=probs,
                mode="lines+markers",
                name="Win probability",
                line=dict(width=3, color="#22d3ee"),
                marker=dict(size=6),
            )
        )
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="rgba(255,255,255,.35)",
            annotation_text="50% baseline",
        )
        fig.update_layout(
            title=f"Dynamic Win Probability · Match {match_id}",
            height=440,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe5f3"),
            yaxis=dict(
                title="Win probability (%)",
                range=[0, 100],
                gridcolor="rgba(255,255,255,.07)",
            ),
            xaxis=dict(title="Overs"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        card("Target", str(match_target), "first innings score + 1")
        card("Outcome", "WON" if match_won else "LOST", "synthetic chase result")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            match_data[
                [
                    "over",
                    "runs",
                    "wickets",
                    "current_run_rate",
                    "required_run_rate",
                    "final_score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 5 — NOTES
# ============================================================
with tabs[4]:
    st.markdown('<div class="section-title">Project Notes</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="prob-card">
            <div style="font-size:1.15rem;font-weight:900;color:#fff;">What this dashboard demonstrates</div>
            <div style="margin-top:10px;color:#a8b5c7;line-height:1.75;">
                <b style="color:#eaf2ff;">Machine Learning:</b> supervised regression for final-score prediction.<br>
                <b style="color:#eaf2ff;">Polynomial Curve Fitting:</b> degree-2 and degree-3 polynomial regression.<br>
                <b style="color:#eaf2ff;">Discrete Random Variables:</b> empirical runs-per-over PMF, expectation and variance.<br>
                <b style="color:#eaf2ff;">Bayes Rule:</b> P(Win | Ahead of Required Run Rate).<br>
                <b style="color:#eaf2ff;">Statistics:</b> mean, quantiles, covariance and score distribution.<br>
                <b style="color:#eaf2ff;">Interactive UI:</b> live-style match inputs and visual analytics in Streamlit.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">SOURCE MODEL</div>
                <div style="color:#fff;font-size:1rem;font-weight:850;margin-top:8px;">
                    Uploaded notebook logic preserved
                </div>
                <div class="metric-note">
                    Synthetic 200-match generator · seed 42 · 6 input features ·
                    Linear + Polynomial degree 2/3.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">IMPORTANT LIMITATION</div>
                <div style="color:#fff;font-size:1rem;font-weight:850;margin-top:8px;">
                    Academic prototype, not a live betting engine
                </div>
                <div class="metric-note">
                    The notebook uses synthetic data. For real-world accuracy,
                    replace it with licensed or public ball-by-ball historical data
                    and retrain the models.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer-note">
            CRICKETIQ · Academic ML & Probability project · Built with Streamlit, scikit-learn, Plotly, NumPy and Pandas.
        </div>
        """,
        unsafe_allow_html=True,
    )
