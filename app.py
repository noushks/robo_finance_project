# -*- coding: utf-8 -*-
"""
RoboPlanner: a simple robo-advisor, Streamlit version (FINC6580 Fintech, Week 3)
===============================================================================

RUN LOCALLY:   pip install -r requirements.txt   then   streamlit run app.py
DEPLOY:        push app.py, requirements.txt and the .streamlit folder to a GitHub repo, then
               on share.streamlit.io click "Create app" and choose app.py.

Same model as the Colab notebook: goal bucket (cash/bonds for goals within 5 years) plus a
growth bucket (110 - age in stocks, tilted by a 4-question risk quiz), projected with
1,000 Monte Carlo scenarios in today's dollars.

AI PROMPTS USED TO BUILD THIS (keep this log for the Canvas submission)
  Prompt 1: "What would be the best steps to start this project? Give me the steps
             (not the code). I think I would like to design a streamlit app for the
             assignment - would that be feasible?"
  Prompt 2: "Create a google colab friendly code file (python). Keep it simple, and add
             elements that make the application look clean and easy to use. Use pastel
             colors wherever applicable"
  Prompt 3: "Can this app be opened in streamlit.io?"
  Prompt 4: "Yes, build the streamlit version of the app"
  Prompt 5: "just for future reference, this is the link to my repo: https://github.com/noushks/robo_finance_project"
  (Add any follow-up prompts you use here.)

REPOSITORY: https://github.com/noushks/robo_finance_project

DISCLAIMER: Educational project only, not financial advice.
"""

# %% 1. Setup 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# %% 2. Assumptions
INFLATION = 0.037       # annual inflation of 3.7% in the US
INCOME_GROWTH = 0.03     # annual nominal raise
SAFE_RATE = 0.035        # discount rate for money set aside for near-term goals
EARMARK_YEARS = 5        # goals this close get their own safe bucket
US_SHARE = 0.60          # share of the stock portion in US stocks (rest international)
N_SIMS = 1000

ASSETS = pd.DataFrame(
    {
        "ticker":     ["VTI", "VXUS", "BND", "SGOV"],
        "exp_return": [0.070, 0.070, 0.040, 0.030],   # nominal expected returns
        "volatility": [0.160, 0.170, 0.060, 0.010],
    },
    index=["US stocks", "International stocks", "Bonds", "Cash / T-bills"],
)
CORR = np.array([[1.0, 0.8, 0.1, 0.0],
                 [0.8, 1.0, 0.1, 0.0],
                 [0.1, 0.1, 1.0, 0.1],
                 [0.0, 0.0, 0.1, 1.0]])
MU = ASSETS["exp_return"].values
COV = np.outer(ASSETS["volatility"], ASSETS["volatility"]) * CORR

# Pastel palette
P = {
    "lavender": "#CDB4DB", "pink": "#FFC8DD", "rose": "#FFAFCC", "blue": "#BDE0FE",
    "sky": "#A2D2FF", "mint": "#B9FBC0", "peach": "#FFD6A5", "butter": "#FDFFB6",
    "ink": "#3D405B", "muted": "#7A7D99", "bg": "#FBF8FF", "deep": "#8E7DBE",
}
ASSET_COLORS = [P["sky"], P["lavender"], P["mint"], P["peach"]]

# 3. Risk quiz
QUIZ = [
    ("If your investments fell 20% in one month, you would…",
     [("Sell everything", 1), ("Sell some", 2), ("Hold and wait", 3), ("Buy more", 4)]),
    ("Your main investing priority is…",
     [("Protect what I have", 1), ("Steady income", 2), ("Balanced growth", 3), ("Maximum growth", 4)]),
    ("How experienced are you with investing?",
     [("None", 1), ("A little", 2), ("Comfortable", 3), ("Very experienced", 4)]),
    ("Which one-year range of outcomes could you live with?",
     [("-5% to +8%", 1), ("-12% to +15%", 2), ("-20% to +25%", 3), ("-30% to +35%", 4)]),
]


def risk_profile(points):
    """Score 4-16 -> label and a tilt (percentage points of stocks, -18 to +18)."""
    score = sum(points)
    tilt = (score - 10) * 3
    if score <= 6:
        label = "Conservative"
    elif score <= 9:
        label = "Moderately conservative"
    elif score <= 11:
        label = "Moderate"
    elif score <= 13:
        label = "Moderately aggressive"
    else:
        label = "Aggressive"
    return score, label, tilt


# Allocation rules
def mix(equity, cash):
    """Turn a stock share and a cash share into weights for the 4 asset classes."""
    equity = float(np.clip(equity, 0, 1))
    cash = float(np.clip(cash, 0, 1 - equity))
    return np.array([equity * US_SHARE, equity * (1 - US_SHARE), 1 - equity - cash, cash])


def growth_mix(age, retire_age, tilt):
    """Long-term money: 110 - age in stocks, tilted by risk, kept between 20% and 95%."""
    equity = np.clip(110 - age + tilt, 20, 95) / 100
    cash = 0.05 if age >= retire_age else 0.0      # retirees keep a small cash buffer
    return mix(equity, cash)


def goal_mix(years_away):
    """Near-term goal money: very safe if within 2 years, mostly bonds if 3-5 years."""
    return mix(0.0, 0.60) if years_away <= 2 else mix(0.25, 0.15)


# Simulation
def simulate(p, deterministic=False, n_sims=N_SIMS, seed=7):
    """Year-by-year projection. Money flows in at the start of each year, then grows."""
    _, _, tilt = risk_profile(p["risk"])
    T = max(p["plan_age"] - p["age"], 1)
    infl = (1 + INFLATION) ** np.arange(T + 1)

    contrib, spend = np.zeros(T), np.zeros(T)
    for t in range(T):
        a = p["age"] + t
        if a < p["retire_age"]:
            rate = p["save_now"] if a < p["switch_age"] else p["save_later"]
            contrib[t] = p["income"] * (1 + INCOME_GROWTH) ** t * rate
        else:
            spend[t] = max(p["ret_spend"] - p["other_income"], 0) * infl[t]

    goals = [(g["name"], g["years"], g["amount"] * infl[g["years"]])
             for g in p["goals"] if 1 <= g["years"] < T]

    n = 1 if deterministic else n_sims
    if deterministic:
        R = np.tile(MU, (1, T, 1))
    else:
        R = np.random.default_rng(seed).multivariate_normal(MU, COV, size=(n, T))
    R = np.maximum(R, -0.95)

    B = np.zeros((n, T + 1))
    B[:, 0] = p["assets"]
    spend_short = np.zeros(n, bool)
    goal_ok = [np.ones(n, bool) for _ in goals]
    W = np.zeros((T, 4))

    for t in range(T):
        bal = B[:, t] + contrib[t]
        for i, (_, y, amt) in enumerate(goals):
            if y == t:
                goal_ok[i] = bal >= amt * 0.999
                bal = bal - amt
        bal = bal - spend[t]
        spend_short |= bal < -1
        bal = np.maximum(bal, 0)

        # Split today's balance into the goal bucket and the growth bucket
        safe_dollars, earmark = np.zeros(4), 0.0
        for _, y, amt in goals:
            k = y - t
            if 1 <= k <= EARMARK_YEARS:
                pv = amt / (1 + SAFE_RATE) ** k
                earmark += pv
                safe_dollars += pv * goal_mix(k)
        g_w = growth_mix(p["age"] + t, p["retire_age"], tilt)
        s_w = safe_dollars / earmark if earmark > 0 else g_w
        frac = np.where(bal > 0, np.minimum(earmark, bal) / np.maximum(bal, 1e-9), 0.0)
        weights = frac[:, None] * s_w + (1 - frac[:, None]) * g_w

        B[:, t + 1] = bal * (1 + (weights * R[:, t, :]).sum(axis=1))
        W[t] = weights[0]

    real = B / infl
    success = ~spend_short
    for ok in goal_ok:
        success &= ok
    return {
        "ages": p["age"] + np.arange(T + 1), "B": B, "real": real, "W": W,
        "goals": goals, "goal_prob": [ok.mean() for ok in goal_ok],
        "success": success.mean(), "spend": spend, "contrib": contrib,
        "bequest_prob": (real[:, -1] >= p["bequest"]).mean() if p["bequest"] > 0 else None,
    }


def current_buckets(p):
    """Today's split between goal buckets and the growth bucket (for the table)."""
    _, _, tilt = risk_profile(p["risk"])
    rows, used = [], 0.0
    for g in sorted(p["goals"], key=lambda g: g["years"]):
        if g["years"] <= EARMARK_YEARS:
            need = g["amount"] * (1 + INFLATION) ** g["years"] / (1 + SAFE_RATE) ** g["years"]
            have = min(need, max(p["assets"] - used, 0))
            used += have
            rows.append((f"🎯 {g['name']} ({g['years']} yrs)", have, goal_mix(g["years"]), need))
    rows.append(("🌱 Long-term growth", max(p["assets"] - used, 0),
                 growth_mix(p["age"], p["retire_age"], tilt), None))
    return rows


# Helpers for display 
def money(x):
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


def style_fig(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16, color=P["ink"])),
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Nunito, Segoe UI, Helvetica, Arial", color=P["ink"]),
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.18, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E6E1F0")
    fig.update_yaxes(gridcolor="#F1EDF7", zeroline=False)
    return fig


def donut(weights, center_text):
    keep = weights > 0.005                       # hide empty slices
    fig = go.Figure(go.Pie(
        labels=[f"{n} ({t})" for n, t, k in zip(ASSETS.index, ASSETS["ticker"], keep) if k],
        values=np.round(weights[keep] * 100, 1), hole=0.6, sort=False,
        marker=dict(colors=[c for c, k in zip(ASSET_COLORS, keep) if k], line=dict(color="white", width=3)),
        textinfo="percent", hovertemplate="%{label}: %{value:.0f}%<extra></extra>",
    ))
    fig.add_annotation(text=center_text, showarrow=False, font=dict(size=15, color=P["ink"]))
    return style_fig(fig, "Recommended mix today", 360)


def fan_chart(sim, det, p):
    ages, real = sim["ages"], sim["real"]
    pc = {q: np.percentile(real, q, axis=0) for q in (10, 25, 50, 75, 90)}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=pc[90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pc[10], fill="tonexty", fillcolor="rgba(205,180,219,0.35)",
                             line=dict(width=0), name="10th–90th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pc[75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pc[25], fill="tonexty", fillcolor="rgba(205,180,219,0.7)",
                             line=dict(width=0), name="25th–75th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=pc[50], line=dict(color=P["deep"], width=3), name="Median",
                             hovertemplate="Age %{x}: %{y:$,.0f}<extra>Median</extra>"))
    fig.add_trace(go.Scatter(x=ages, y=det["real"][0], line=dict(color="#F4978E", width=2, dash="dot"),
                             name="Expected-return path",
                             hovertemplate="Age %{x}: %{y:$,.0f}<extra>Expected</extra>"))
    if p["age"] < p["retire_age"] < p["plan_age"]:
        fig.add_vline(x=p["retire_age"], line=dict(color=P["peach"], width=3, dash="dash"))
        fig.add_annotation(x=p["retire_age"], y=1.02, yref="paper", text="🌴 Retire",
                           showarrow=False, font=dict(color=P["muted"]))
    for name, y, _ in sim["goals"]:
        fig.add_vline(x=p["age"] + y, line=dict(color=P["rose"], width=2, dash="dot"))
        fig.add_annotation(x=p["age"] + y, y=0.95, yref="paper", text=f"🎯 {name}",
                           showarrow=False, font=dict(color=P["muted"], size=11), xanchor="left")
    fig.update_yaxes(tickprefix="$", tickformat="~s", title="Portfolio value (today's $)")
    fig.update_xaxes(title="Age")
    return style_fig(fig, "Projected portfolio: 1,000 simulated markets", 430)


def glide_chart(det):
    ages = det["ages"][:-1]
    fig = go.Figure()
    for i, name in enumerate(ASSETS.index):
        fig.add_trace(go.Scatter(x=ages, y=det["W"][:, i] * 100, stackgroup="one", name=name,
                                 line=dict(width=0.5, color=ASSET_COLORS[i]),
                                 fillcolor=ASSET_COLORS[i],
                                 hovertemplate=f"{name}: %{{y:.0f}}%<extra></extra>"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_xaxes(title="Age")
    return style_fig(fig, "How the mix shifts over time (glide path)", 340)


def metric_card(title, value, sub, color):
    return (f"<div class='rp-metric' style='background:{color}'>"
            f"<div class='rp-mt'>{title}</div><div class='rp-mv'>{value}</div>"
            f"<div class='rp-ms'>{sub}</div></div>")


def advice(p, sim, buckets, label):
    tips = []
    retired = p["age"] >= p["retire_age"]
    for (name, have, mixw, need) in buckets[:-1]:
        gap = need - have
        tips.append(f"Keep <b>{money(have)}</b> for <b>{name[2:]}</b> in cash and short-term bonds "
                    f"(~{mixw[3]*100:.0f}% cash, {mixw[2]*100:.0f}% bonds). Money you need soon "
                    f"shouldn't ride the stock market.")
        if gap > 1:
            tips.append(f"Today's assets cover {have/need*100:.0f}% of that goal. Your planned "
                        f"savings need to fill a gap of about <b>{money(gap)}</b> (today's value).")
    for g in p["goals"]:
        if g["years"] > EARMARK_YEARS:
            tips.append(f"<b>{g['name']}</b> is {g['years']} years away, so it stays in the growth bucket "
                        f"for now. Start moving it to safety about {EARMARK_YEARS} years before you need it.")
    gw = buckets[-1][2]
    eq = (gw[0] + gw[1]) * 100
    tips.append(f"Invest long-term money about <b>{eq:.0f}% stocks / {100-eq:.0f}% bonds & cash</b>, "
                f"in line with your <b>{label.lower()}</b> risk profile. Shift roughly 1% a year toward bonds as you age.")
    if retired:
        net = max(p["ret_spend"] - p["other_income"], 0)
        wr = net / p["assets"] * 100 if p["assets"] else 0
        flag = "a sustainable level" if wr <= 4.5 else "on the high side; consider trimming spending"
        tips.append(f"Your portfolio needs to supply <b>{money(net)}/yr</b>, an initial withdrawal "
                    f"rate of <b>{wr:.1f}%</b>, which is {flag}.")
    else:
        idx = p["retire_age"] - p["age"]
        if 0 < idx < len(sim["ages"]):
            med = np.median(sim["real"][:, idx])
            tips.append(f"At {p['retire_age']} your median balance is about <b>{money(med)}</b> (today's $), "
                        f"enough for roughly <b>{money(0.04*med + p['other_income'])}/yr</b> "
                        f"using the 4% rule plus other income.")
    s = sim["success"] * 100
    if s < 70:
        tips.append("⚠️ Success odds are low. Try saving more, retiring a little later, or spending less in "
                    "retirement, then rebuild the plan to compare.")
    elif s < 85:
        tips.append("👍 You're close. A small boost in savings or a year or two of extra work would add a safety margin.")
    else:
        tips.append("🎉 You're on track in most market scenarios. Review the plan yearly or after big life changes.")
    if sim["bequest_prob"] is not None:
        tips.append(f"Chance of leaving at least <b>{money(p['bequest'])}</b> (today's $) at age "
                    f"{p['plan_age']}: <b>{sim['bequest_prob']*100:.0f}%</b>.")
    return "".join(f"<li>{t}</li>" for t in tips)



# The Streamlit app
st.set_page_config(page_title="RoboPlanner", layout="wide")

st.markdown(f"""
<style>
.rp-banner {{background: linear-gradient(120deg, {P['pink']}, {P['lavender']}, {P['blue']});
  border-radius: 18px; padding: 22px 26px; color: {P['ink']}; margin-bottom: 14px;}}
.rp-banner h1 {{margin: 0; font-size: 30px; color: {P['ink']}; padding: 0;}}
.rp-banner p {{margin: 6px 0 0; font-size: 15px;}}
.rp-metrics {{display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 14px;}}
.rp-metric {{flex: 1 1 170px; border-radius: 16px; padding: 14px 16px; color: {P['ink']};}}
.rp-mt {{font-size: 13px; opacity: .8;}} .rp-mv {{font-size: 24px; font-weight: 800; margin: 2px 0;}}
.rp-ms {{font-size: 12px; opacity: .75;}}
.rp-advice {{background: {P['bg']}; border-left: 6px solid {P['lavender']}; border-radius: 12px;
  padding: 12px 20px; color: {P['ink']}; line-height: 1.6;}}
.rp-table {{border-collapse: collapse; width: 100%; color: {P['ink']}; font-size: 14px;}}
.rp-table th {{background: {P['lavender']}; padding: 8px; text-align: left;}}
.rp-table td {{padding: 8px; border-bottom: 1px solid #EEE7F7;}}
.rp-table tr:nth-child(even) td {{background: {P['bg']};}}
.rp-note {{color: {P['muted']}; font-size: 13px;}}
div[data-testid="stTabs"] {{background: {P['bg']}; border-radius: 16px; padding: 6px 18px 12px;
  border: 1px solid #EEE7F7;}}
div.stButton > button {{border-radius: 12px; border: none; font-weight: 600; color: {P['ink']};
  background: {P['blue']}; box-shadow: 0 2px 6px rgba(61,64,91,0.12);}}
div.stButton > button:hover {{background: {P['lavender']}; color: {P['ink']};}}
</style>
""", unsafe_allow_html=True)

# ---- Defaults and sample cases (stored in session_state so buttons can fill the form) ----
GOAL_DEFAULTS = [("House down payment", 60_000, 2), ("College fund", 100_000, 15), ("Other goal", 25_000, 5)]
DEFAULTS = dict(name="", age=30, retire=65, plan=95, assets=50_000, income=80_000, save_now=15,
                save_later=15, switch=40, spend=60_000, other=20_000, bequest=0, risk=[3, 3, 3, 3])
CASES = {
    "A": dict(name="Case A: 30-year-old saver", age=30, retire=65, plan=95, assets=50_000, income=85_000,
              save_now=20, save_later=15, switch=32, spend=60_000, other=20_000, bequest=0,
              goals=[("House down payment", 80_000, 2)], risk=[3, 4, 3, 3]),
    "B": dict(name="Case B: 55-year-old", age=55, retire=65, plan=95, assets=200_000, income=100_000,
              save_now=15, save_later=15, switch=55, spend=55_000, other=25_000, bequest=0,
              goals=[], risk=[3, 3, 3, 2]),
    "C": dict(name="Case C: 70-year-old retiree", age=70, retire=70, plan=95, assets=2_000_000, income=0,
              save_now=0, save_later=0, switch=70, spend=110_000, other=35_000, bequest=500_000,
              goals=[], risk=[2, 1, 3, 2]),
}


def apply_values(c):
    ss = st.session_state
    for k in ["name", "age", "retire", "plan", "assets", "income", "save_now", "save_later",
              "switch", "spend", "other", "bequest"]:
        ss[k] = c[k]
    goals = c.get("goals", [])
    for i, d in enumerate(GOAL_DEFAULTS):
        on = i < len(goals)
        g = goals[i] if on else d
        ss[f"g_on{i}"], ss[f"g_nm{i}"], ss[f"g_amt{i}"], ss[f"g_yrs{i}"] = on, g[0], g[1], g[2]
    for i, pts in enumerate(c["risk"]):
        ss[f"q{i}"] = pts


if "age" not in st.session_state:
    apply_values(DEFAULTS)

# ---- Header and sample-case buttons ----
st.markdown("<div class='rp-banner'><h1>🌸 RoboPlanner</h1>"
            "<p>A friendly robo-advisor: answer a few questions and get a personalised investment plan.</p>"
            "</div>", unsafe_allow_html=True)

st.markdown("<div class='rp-note'>Try a sample case, or enter your own details in each of the tabs. "
            "The plan updates as you type.</div>", unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
b1.button("🏠 Case A: 30-yr-old", on_click=apply_values, args=(CASES["A"],), width="stretch")
b2.button("💼 Case B: 55-yr-old", on_click=apply_values, args=(CASES["B"],), width="stretch")
b3.button("🌴 Case C: 70-yr-old", on_click=apply_values, args=(CASES["C"],), width="stretch")
b4.button("↺ Start fresh", on_click=apply_values, args=(DEFAULTS,), width="stretch")

# ---- Inputs ----
t1, t2, t3, t4 = st.tabs([" About you", "Your money", "Goals", "Risk quiz"])
with t1:
    c1, c2 = st.columns(2)
    c1.text_input("Name (optional)", key="name", placeholder="e.g. Alex")
    c1.slider("Current age", 18, 95, key="age")
    c2.slider("Retirement age", 40, 85, key="retire")
    c2.slider("Plan until age (life expectancy)", 70, 105, key="plan")
with t2:
    st.caption("Use today's dollars. Savings can change later (e.g. after buying a house).")
    c1, c2 = st.columns(2)
    c1.number_input("Current investments ($)", 0, 100_000_000, step=5_000, key="assets")
    c1.number_input("Annual income before tax ($)", 0, 10_000_000, step=5_000, key="income")
    c1.number_input("Retirement spending per year (today's $)", 0, 10_000_000, step=5_000, key="spend")
    c1.number_input("Pension / Social Security per year ($)", 0, 1_000_000, step=1_000, key="other")
    c2.slider("Savings rate now (% of income)", 0, 60, key="save_now")
    c2.slider("Savings rate later (%)", 0, 60, key="save_later")
    c2.slider("…switching at age", 18, 85, key="switch")
with t3:
    st.caption("Tick the goals that apply. Amounts are in US dollars.")
    for i in range(3):
        c0, c1, c2, c3 = st.columns([0.6, 2.2, 1.6, 2.2])
        c0.checkbox("On", key=f"g_on{i}")
        c1.text_input("Goal", key=f"g_nm{i}")
        c2.number_input("Amount ($)", 0, 50_000_000, step=5_000, key=f"g_amt{i}")
        c3.slider("Years from now", 1, 40, key=f"g_yrs{i}")
    st.number_input("Leave to heirs/bequest (today's $)", 0, 100_000_000, step=50_000, key="bequest")
with t4:
    st.caption("Four quick questions to gauge how much ups and downs you can handle.")
    for i, (q, opts) in enumerate(QUIZ):
        labels = {pts: lab for lab, pts in opts}
        st.radio(f"{i+1}. {q}", [1, 2, 3, 4], format_func=labels.get, horizontal=True, key=f"q{i}")

ss = st.session_state
p = dict(
    name=ss.name.strip(), age=ss.age, retire_age=ss.retire, plan_age=max(ss.plan, ss.age + 1),
    assets=ss.assets, income=ss.income, save_now=ss.save_now / 100, save_later=ss.save_later / 100,
    switch_age=ss.switch, ret_spend=ss.spend, other_income=ss.other, bequest=ss.bequest,
    goals=[dict(name=ss[f"g_nm{i}"] or "Goal", amount=ss[f"g_amt{i}"], years=ss[f"g_yrs{i}"])
           for i in range(3) if ss[f"g_on{i}"] and ss[f"g_amt{i}"] > 0],
    risk=[ss[f"q{i}"] for i in range(4)],
)


@st.cache_data(show_spinner=False)
def run_model(p):
    return simulate(p), simulate(p, deterministic=True), current_buckets(p)


# ---- Results ----
with st.spinner(f"Running {N_SIMS:,} market scenarios…"):
    sim, det, buckets = run_model(p)

score, label, _ = risk_profile(p["risk"])
total_w = sum(b[1] * b[2] for b in buckets) / max(sum(b[1] for b in buckets), 1)
eq_today = (total_w[0] + total_w[1]) * 100
lt_eq = (buckets[-1][2][0] + buckets[-1][2][1]) * 100
retired = p["age"] >= p["retire_age"]
idx = 0 if retired else min(p["retire_age"] - p["age"], len(sim["ages"]) - 1)
med_ret, med_end = np.median(sim["real"][:, idx]), np.median(sim["real"][:, -1])
s = sim["success"] * 100
s_color = P["mint"] if s >= 85 else (P["butter"] if s >= 70 else P["rose"])
who = f"{p['name']}'s plan" if p["name"] else "Your plan"

st.subheader(f"📋 {who}")
st.markdown("<div class='rp-metrics'>"
    + metric_card("Risk profile", label, f"Quiz score {score}/16", P["lavender"])
    + metric_card("Long-term mix", f"{lt_eq:.0f}% stocks", f"all money today: {eq_today:.0f}% stocks", P["blue"])
    + metric_card("At retirement (median)" if not retired else "Portfolio today",
                  money(med_ret), "today's dollars", P["peach"])
    + metric_card("Plan success odds", f"{s:.0f}%", f"goals met & money lasts to {p['plan_age']}", s_color)
    + metric_card(f"Left at age {p['plan_age']} (median)", money(med_end), "today's dollars", P["pink"])
    + "</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1.25])
with left:
    st.plotly_chart(donut(total_w, f"<b>{eq_today:.0f}%</b><br>stocks"), theme=None)
with right:
    rows = "".join(
        f"<tr><td>{n}</td><td>{money(amt)}</td><td>{(m[0]+m[1])*100:.0f}%</td>"
        f"<td>{m[2]*100:.0f}%</td><td>{m[3]*100:.0f}%</td></tr>" for n, amt, m, _ in buckets)
    st.markdown("#### 🪣 Where today's money goes")
    st.markdown("<table class='rp-table'><tr><th>Bucket</th><th>Amount</th><th>Stocks</th><th>Bonds</th>"
                f"<th>Cash</th></tr>{rows}</table>"
                "<div class='rp-note' style='margin-top:6px'>Example funds: VTI (US stocks), VXUS "
                "(international), BND (bonds), SGOV (T-bills). Any low-cost equivalents work.</div>",
                unsafe_allow_html=True)
    if sim["goals"]:
        g_rows = "".join(
            f"<tr><td>🎯 {n}</td><td>age {p['age']+y}</td><td>{money(a/(1+INFLATION)**y)}</td>"
            f"<td>{pr*100:.0f}%</td></tr>" for (n, y, a), pr in zip(sim["goals"], sim["goal_prob"]))
        st.markdown("#### 🎯 Goal check")
        st.markdown("<table class='rp-table'><tr><th>Goal</th><th>When</th><th>Amount (today's $)</th>"
                    f"<th>Chance fully funded</th></tr>{g_rows}</table>", unsafe_allow_html=True)

st.plotly_chart(fan_chart(sim, det, p), theme=None)
st.plotly_chart(glide_chart(det), theme=None)

st.markdown("#### What RoboPlanner recommends")
st.markdown(f"<div class='rp-advice'><ul>{advice(p, sim, buckets, label)}</ul></div>",
            unsafe_allow_html=True)

st.write("")
with st.expander("Assumptions"):
    st.markdown(f"""
- Expected returns (nominal): stocks 7%, bonds 4%, cash 3%; volatility 16–17% / 6% / 1%.
- Inflation {INFLATION*100:.1f}%, raises {INCOME_GROWTH*100:.0f}%/yr. Charts are in today's US dollars.
- Long-term mix = 110 − age in stocks, ± up to 18 points from the risk quiz (bounds 20%–95%).
- Goals within {EARMARK_YEARS} years are held in cash and short-term bonds.
- {N_SIMS:,} Monte Carlo scenarios with correlated annual returns. Before-tax, no fees.
""")
with st.expander("Ideas I would like to include for the future (potential V2)"):
    st.markdown("""
- Life expectancy tables by sex and health, instead of a single "plan until" age.
- Marital status, a spouse's income and benefits, and dependents.
- Taxes and account types (401(k)/RRSP, Roth/TFSA, taxable) and withdrawal ordering.
- Home equity, mortgages and other real estate; debt payoff.
- Social Security/CPP claiming-age optimisation, annuities, fees and rebalancing rules.
""")
st.caption("Educational tool only, not financial advice.")
