# -*- coding: utf-8 -*-
"""
RoboPlanner: a simple robo-advisor  (FINC6580 Fintech, Week 3: "Build your own robo")
==================================================================================

HOW TO RUN IN GOOGLE COLAB
  Option 1: Upload the .ipynb version and choose Runtime > Run all.
  Option 2: Open a new Colab notebook, paste this whole file into ONE cell, and run it.
  The app appears below the cell: fill in the tabs (or click a sample case) and press
  "Build my plan".

WHAT IT DOES
  1. Gathers information: age, assets, income, savings rates (now and later), goals
     with dates, retirement spending, other retirement income, and a 4-question risk quiz.
  2. Maps it to a recommendation:
       - Goal bucket: money needed within 5 years sits in cash and short-term bonds.
       - Growth bucket: everything else follows an age-based glide path
         (110 - age in stocks), tilted up or down by the risk-quiz score.
  3. Displays the plan: allocation donut with example ETFs, a Monte Carlo projection
     (1,000 scenarios, in today's dollars), the glide path over time, goal success
     odds, and plain-English advice.

AI PROMPTS USED TO BUILD THIS (keep this log for the Canvas submission)
  Prompt 1: "What would be the best steps to start this project? Give me the steps
             (not the code). I think I would like to design a streamlit app for the
             assignment - would that be feasible?"
  Prompt 2: "Create a google colab friendly code file (python). Keep it simple, and add
             elements that make the application look clean and easy to use. Use pastel
             colors wherever applicable"
  (Add any follow-up prompts you use here.)

REPOSITORY: https://github.com/noushks/robo_finance_project

DISCLAIMER: Educational project only, not financial advice.
"""

# %% 1. Setup ------------------------------------------------------------------------
import numpy as np
import pandas as pd
import ipywidgets as w
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display, HTML, clear_output

try:
    import google.colab  # noqa: F401
    pio.renderers.default = "colab"
except ImportError:
    pass

# %% 2. Assumptions (edit these to change the model) ----------------------------------
INFLATION = 0.025        # annual inflation
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

# %% 3. Risk quiz ----------------------------------------------------------------------
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


# %% 4. Allocation rules -------------------------------------------------------------
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


# %% 5. Simulation ---------------------------------------------------------------------
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


# %% 6. Helpers for display ------------------------------------------------------------
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


# %% 7. The app (widgets) ---------------------------------------------------------------
CSS = f"""
<style>
.rp-banner {{background: linear-gradient(120deg, {P['pink']}, {P['lavender']}, {P['blue']});
  border-radius: 18px; padding: 22px 26px; color: {P['ink']}; font-family: Nunito, Segoe UI, sans-serif;}}
.rp-banner h1 {{margin: 0; font-size: 28px;}} .rp-banner p {{margin: 6px 0 0; font-size: 14px;}}
.rp-card {{background: {P['bg']}; border-radius: 16px; padding: 14px 18px !important;
  border: 1px solid #EEE7F7; box-shadow: 0 2px 8px rgba(142,125,190,0.10);}}
.rp-h {{font-family: Nunito, Segoe UI, sans-serif; color: {P['ink']}; font-weight: 700; font-size: 15px; margin: 4px 0;}}
.rp-note {{font-family: Nunito, Segoe UI, sans-serif; color: {P['muted']}; font-size: 12.5px;}}
.rp-btn {{border-radius: 12px !important; color: {P['ink']} !important; font-weight: 600 !important;
  border: none !important; box-shadow: 0 2px 6px rgba(61,64,91,0.12) !important;}}
.rp-metrics {{display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0;}}
.rp-metric {{flex: 1 1 170px; border-radius: 16px; padding: 14px 16px; font-family: Nunito, Segoe UI, sans-serif; color: {P['ink']};}}
.rp-mt {{font-size: 12.5px; opacity: .8;}} .rp-mv {{font-size: 24px; font-weight: 800; margin: 2px 0;}}
.rp-ms {{font-size: 12px; opacity: .75;}}
.rp-advice {{background: {P['bg']}; border-left: 6px solid {P['lavender']}; border-radius: 12px;
  padding: 12px 20px; font-family: Nunito, Segoe UI, sans-serif; color: {P['ink']}; line-height: 1.55;}}
.rp-table {{border-collapse: collapse; width: 100%; font-family: Nunito, Segoe UI, sans-serif; color: {P['ink']}; font-size: 13.5px;}}
.rp-table th {{background: {P['lavender']}; padding: 8px; text-align: left;}}
.rp-table td {{padding: 8px; border-bottom: 1px solid #EEE7F7;}}
.rp-table tr:nth-child(even) td {{background: {P['bg']};}}
details.rp-details {{font-family: Nunito, Segoe UI, sans-serif; color: {P['ink']}; background: {P['butter']}55;
  border-radius: 12px; padding: 10px 16px; margin-top: 10px;}}
</style>
"""

ST = {"description_width": "230px"}
WIDE = w.Layout(width="520px")


def money_box(desc, value, step=1000):
    return w.BoundedIntText(value=value, min=0, max=100_000_000, step=step, description=desc,
                            style=ST, layout=WIDE)


def pct_slider(desc, value, maxv=60):
    return w.IntSlider(value=value, min=0, max=maxv, description=desc, style=ST, layout=WIDE,
                       readout_format="d")


def btn(text, color):
    b = w.Button(description=text, layout=w.Layout(width="auto", height="38px", margin="4px"))
    b.style.button_color = color
    b.add_class("rp-btn")
    return b


def section(title, note, *kids):
    box = w.VBox([w.HTML(f"<div class='rp-h'>{title}</div><div class='rp-note'>{note}</div>"), *kids])
    box.add_class("rp-card")
    return box


# Inputs: about you
name_in = w.Text(value="", placeholder="e.g. Alex", description="Name (optional)", style=ST, layout=WIDE)
age_in = w.IntSlider(value=30, min=18, max=95, description="Current age", style=ST, layout=WIDE)
ret_in = w.IntSlider(value=65, min=40, max=85, description="Retirement age", style=ST, layout=WIDE)
plan_in = w.IntSlider(value=95, min=70, max=105, description="Plan until age (life expectancy)",
                      style=ST, layout=WIDE)

# Inputs: finances
assets_in = money_box("Current investments ($)", 50_000)
income_in = money_box("Annual income before tax ($)", 80_000)
save_now_in = pct_slider("Savings rate now (% of income)", 15)
save_later_in = pct_slider("Savings rate later (%)", 15)
switch_in = w.IntSlider(value=40, min=18, max=85, description="…switching at age", style=ST, layout=WIDE)
spend_in = money_box("Retirement spending / yr (today's $)", 60_000)
other_in = money_box("Pension / Social Security / yr ($)", 20_000)

# Inputs: goals (up to 3) and bequest
goal_rows = []
for default in [("House down payment", 60_000, 2), ("College fund", 100_000, 15), ("Other goal", 25_000, 5)]:
    on = w.Checkbox(value=False, description="", indent=False, layout=w.Layout(width="30px"))
    nm = w.Text(value=default[0], layout=w.Layout(width="190px"))
    amt = w.BoundedIntText(value=default[1], min=0, max=50_000_000, step=1000, description="$",
                           style={"description_width": "15px"}, layout=w.Layout(width="150px"))
    yrs = w.IntSlider(value=default[2], min=1, max=40, description="in yrs",
                      style={"description_width": "45px"}, layout=w.Layout(width="230px"))
    goal_rows.append((on, nm, amt, yrs))
bequest_in = money_box("Leave to heirs / bequest (today's $)", 0)

# Inputs: risk quiz
quiz_in = [w.Dropdown(options=opts, value=opts[2][1], description="", layout=w.Layout(width="260px"))
           for _, opts in QUIZ]

tabs = w.Tab(children=[
    section("👤 About you", "Tell us where you are in life.", name_in, age_in, ret_in, plan_in),
    section("💰 Your money", "Use today's dollars. Savings can change later (e.g. after buying a house).",
            assets_in, income_in, save_now_in, save_later_in, switch_in, spend_in, other_in),
    section("🎯 Goals", "Tick the goals that apply. Amounts are in today's dollars.",
            *[w.HBox(list(r)) for r in goal_rows], bequest_in),
    section("🧭 Risk quiz", "Four quick questions to gauge how much ups and downs you can handle.",
            *[w.HBox([w.HTML(f"<div class='rp-note' style='width:330px;font-size:13.5px;color:{P['ink']}'>"
                             f"{i+1}. {q}</div>"), dd]) for i, ((q, _), dd) in enumerate(zip(QUIZ, quiz_in))]),
])
for i, t in enumerate(["👤 About you", "💰 Money", "🎯 Goals", "🧭 Risk quiz"]):
    tabs.set_title(i, t)

# Sample cases from the assignment
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


def load_case(key):
    c = CASES[key]
    name_in.value, age_in.value, ret_in.value, plan_in.value = c["name"], c["age"], c["retire"], c["plan"]
    assets_in.value, income_in.value = c["assets"], c["income"]
    save_now_in.value, save_later_in.value, switch_in.value = c["save_now"], c["save_later"], c["switch"]
    spend_in.value, other_in.value, bequest_in.value = c["spend"], c["other"], c["bequest"]
    for i, (on, nm, amt, yrs) in enumerate(goal_rows):
        if i < len(c["goals"]):
            on.value, (nm.value, amt.value, yrs.value) = True, c["goals"][i]
        else:
            on.value = False
    for dd, pts in zip(quiz_in, c["risk"]):
        dd.value = pts
    build_plan()


def gather():
    return dict(
        name=name_in.value.strip(), age=age_in.value, retire_age=ret_in.value,
        plan_age=max(plan_in.value, age_in.value + 1),
        assets=assets_in.value, income=income_in.value,
        save_now=save_now_in.value / 100, save_later=save_later_in.value / 100, switch_age=switch_in.value,
        ret_spend=spend_in.value, other_income=other_in.value, bequest=bequest_in.value,
        goals=[dict(name=nm.value or "Goal", amount=amt.value, years=yrs.value)
               for on, nm, amt, yrs in goal_rows if on.value and amt.value > 0],
        risk=[dd.value for dd in quiz_in],
    )


out = w.Output()


def build_plan(_=None):
    p = gather()
    with out:
        clear_output(wait=True)
        display(HTML(f"<div class='rp-note'>⏳ Running {N_SIMS:,} market scenarios…</div>"))
        score, label, _ = risk_profile(p["risk"])
        sim, det = simulate(p), simulate(p, deterministic=True)
        buckets = current_buckets(p)
        total_w = sum(b[1] * b[2] for b in buckets) / max(sum(b[1] for b in buckets), 1)
        eq_today = (total_w[0] + total_w[1]) * 100
        lt_eq = (buckets[-1][2][0] + buckets[-1][2][1]) * 100

        retired = p["age"] >= p["retire_age"]
        idx = 0 if retired else min(p["retire_age"] - p["age"], len(sim["ages"]) - 1)
        med_ret = np.median(sim["real"][:, idx])
        med_end = np.median(sim["real"][:, -1])
        s = sim["success"] * 100
        s_color = P["mint"] if s >= 85 else (P["butter"] if s >= 70 else P["rose"])
        who = f"{p['name']}'s plan" if p["name"] else "Your plan"

        clear_output(wait=True)
        display(HTML(f"<div class='rp-h' style='font-size:20px;margin-top:8px'>📋 {who}</div>"))
        display(HTML("<div class='rp-metrics'>"
            + metric_card("Risk profile", label, f"Quiz score {score}/16", P["lavender"])
            + metric_card("Long-term mix", f"{lt_eq:.0f}% stocks", f"all money today: {eq_today:.0f}% stocks", P["blue"])
            + metric_card("At retirement (median)" if not retired else "Portfolio today",
                          money(med_ret), "today's dollars", P["peach"])
            + metric_card("Plan success odds", f"{s:.0f}%",
                          f"goals met & money lasts to {p['plan_age']}", s_color)
            + metric_card(f"Left at age {p['plan_age']} (median)", money(med_end), "today's dollars", P["pink"])
            + "</div>"))

        donut(total_w, f"<b>{eq_today:.0f}%</b><br>stocks").show()

        rows = "".join(
            f"<tr><td>{n}</td><td>{money(amt)}</td><td>{(m[0]+m[1])*100:.0f}%</td>"
            f"<td>{m[2]*100:.0f}%</td><td>{m[3]*100:.0f}%</td></tr>" for n, amt, m, _ in buckets)
        goal_rows_html = "".join(
            f"<tr><td>🎯 {n}</td><td>age {p['age']+y}</td><td>{money(a/(1+INFLATION)**y)}</td>"
            f"<td>{pr*100:.0f}%</td></tr>" for (n, y, a), pr in zip(sim["goals"], sim["goal_prob"]))
        display(HTML(
            "<div class='rp-h'>🪣 Where today's money goes</div>"
            "<table class='rp-table'><tr><th>Bucket</th><th>Amount</th><th>Stocks</th><th>Bonds</th>"
            f"<th>Cash</th></tr>{rows}</table>"
            "<div class='rp-note' style='margin-top:4px'>Example funds: VTI (US stocks), VXUS (international), "
            "BND (bonds), SGOV (T-bills). Any low-cost equivalents work.</div>"
            + ("<div class='rp-h' style='margin-top:14px'>🎯 Goal check</div><table class='rp-table'>"
               "<tr><th>Goal</th><th>When</th><th>Amount (today's $)</th><th>Chance fully funded</th></tr>"
               f"{goal_rows_html}</table>" if sim["goals"] else "")))

        fan_chart(sim, det, p).show()
        glide_chart(det).show()

        display(HTML(f"<div class='rp-h' style='margin-top:6px'>💡 What RoboPlanner recommends</div>"
                     f"<div class='rp-advice'><ul>{advice(p, sim, buckets, label)}</ul></div>"))
        display(HTML(f"""
        <details class='rp-details'><summary><b>📐 Assumptions</b></summary>
        <ul class='rp-note' style='color:{P['ink']}'>
          <li>Expected returns (nominal): stocks 7%, bonds 4%, cash 3%; volatility 16–17% / 6% / 1%.</li>
          <li>Inflation {INFLATION*100:.1f}%, raises {INCOME_GROWTH*100:.0f}%/yr. Charts are in today's dollars.</li>
          <li>Long-term mix = 110 − age in stocks ± up to 18 points from the risk quiz (bounds 20–95%).</li>
          <li>Goals within {EARMARK_YEARS} years are held in cash and short-term bonds.</li>
          <li>{N_SIMS:,} Monte Carlo scenarios with correlated annual returns. Before-tax, no fees.</li>
        </ul></details>
        <details class='rp-details'><summary><b>🚧 Not included yet (ideas for version 2)</b></summary>
        <ul class='rp-note' style='color:{P['ink']}'>
          <li>Life expectancy tables by sex and health, instead of a single "plan until" age.</li>
          <li>Marital status, a spouse's income and benefits, and dependents.</li>
          <li>Taxes and account types (401(k)/RRSP, Roth/TFSA, taxable) and withdrawal ordering.</li>
          <li>Home equity, mortgages and other real estate; debt payoff.</li>
          <li>Social Security/CPP claiming-age optimisation, annuities, fees and rebalancing rules.</li>
        </ul></details>
        <div class='rp-note' style='margin-top:10px'>Educational tool only, not financial advice.</div>"""))


run_btn = btn("✨ Build my plan", P["lavender"])
run_btn.layout = w.Layout(width="220px", height="44px", margin="10px 4px")
run_btn.on_click(build_plan)

case_btns = []
for key, text, color in [("A", "🏠 Case A: 30-yr-old", P["blue"]),
                         ("B", "💼 Case B: 55-yr-old", P["peach"]),
                         ("C", "🌴 Case C: 70-yr-old", P["mint"])]:
    b = btn(text, color)
    b.on_click(lambda _, k=key: load_case(k))
    case_btns.append(b)

display(HTML(CSS))
display(HTML("<div class='rp-banner'><h1>🌸 RoboPlanner</h1>"
             "<p>A friendly robo-advisor: answer a few questions and get a personalised investment plan.</p></div>"))
display(w.VBox([
    w.HTML("<div class='rp-note' style='margin:10px 0 2px'>Try a sample case from the assignment, "
           "or enter your own details below.</div>"),
    w.HBox(case_btns),
    tabs,
    run_btn,
    out,
]))
