# RoboPlanner: a simple robo financial advisor


Repository: https://github.com/noushks/robo_finance_project

RoboPlanner asks for your age, savings, income, savings rates, goals and risk tolerance, then recommends
an investment mix and projects your portfolio across 1,000 simulated markets.

## What's in this repo
| Path | What it is |
|---|---|
| `app.py` | Streamlit app (deploy this on streamlit.io) |
| `requirements.txt` | Python packages the app needs |
| `.streamlit/config.toml` | Pastel colour theme |
| `colab/robo_advisor_colab.ipynb` | Same robo as a Google Colab notebook |
| `colab/robo_advisor_colab.py` | Colab version as a plain Python file |

The AI prompts used to build the app are logged at the top of `app.py` and `colab/robo_advisor_colab.py`.

## How it decides
- **Goal bucket:** money needed within 5 years is held in cash and short-term bonds.
- **Growth bucket:** everything else is invested at about 110 − age in stocks, moved up or down by up to
  18 points based on a 4-question risk quiz.
- **Projection:** 1,000 Monte Carlo scenarios with correlated annual returns, shown in today's dollars.

## Run it
**On your computer**

    pip install -r requirements.txt
    streamlit run app.py

**Online (free):** go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose
`noushks/robo_finance_project`, branch `main`, main file `app.py`, then **Deploy**.

**In Google Colab:** upload `colab/robo_advisor_colab.ipynb` and choose *Runtime → Run all*.

*Educational project only, not financial advice.*
