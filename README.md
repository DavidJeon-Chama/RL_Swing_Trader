# RL-Swing-Trader

A reinforcement learning (PPO) swing-trading agent built for a realistic $100 cash brokerage account — and a set of diagnostic experiments showing **why the agent converged to buy-and-hold**.

> **TL;DR** — Under realistic costs and honest out-of-sample testing, the agent learned that buy-and-hold was hard to beat using price data alone. On unseen 2026 data it averaged **$109.36 per $100** (5 seeds) vs **$113.20** for buy-and-hold. Fixed-policy diagnostics and a rule-based benchmark explained the result: price-only signals showed no edge after trading costs.

---

## Motivation

When I started university, I met a friend working on reinforcement learning for robotics. He told me that using AI is not something to be ashamed of, but a skill engineers need to develop for the future. Knowing that even Wall Street firms use AI for trading, I decided to build my own RL trading agent. My goal was to see whether an agent could learn swing trading under the realistic limits of a $100 personal cash account, and to learn RL by facing its real problems myself.

## Environment design

Custom [Gymnasium](https://gymnasium.farama.org/) environment trained with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

| Component | Design |
| --- | --- |
| Actions | 0 = hold cash / keep position, 1 = buy (all-in), 2 = sell (all-out) |
| Account observations (5) | cash ratio, unsettled cash ratio, in-position flag, unrealized P&L, holding progress |
| Market observations (5) | 5-day momentum, distance from 20-day MA, distance from 60-day MA, drawdown from 60-day high, 20-day volatility |
| Realistic constraints | T+1 settlement, 0.1% fee per trade, 3-day minimum holding period |
| Reward | excess return vs buy-and-hold × 100 + down-day loss penalty + losing-streak penalty on losing sells |
| Universe | 14 U.S. stocks (AAPL, SOFI, F, PEP, MSFT, JNJ, XOM, T, DIS, NKE, WMT, BA, GE, INTC) + KO as a never-trained holdout |

## Evaluation setup

To avoid tuning on the test set, data is split by date:

| Split | Period | Used for |
| --- | --- | --- |
| Train | 2016 (or 2000) – 2022 | learning |
| Validation | 2023 – 2024 | choosing seeds and reward settings |
| Test | 2025 | final model, run once |
| Forward | Jan – Sep 2026 | fully unseen, most honest test |

Every configuration is trained with multiple random seeds, and results are compared to buy-and-hold on both final value and **maximum drawdown (MDD)**.

## Key results

**Forward test (Jan–Sep 2026, unseen by every model), average final value per $100:**

| Strategy | Avg final value | Avg MDD |
| --- | --- | --- |
| Buy-and-hold | $113.20 | −25.7% |
| PPO v6 (5 seeds) | $109.36 | ≈ −25% |
| PPO v5 (5 seeds) | $106.79 | 0% to −25.7% by seed |
| 60-day moving-average rule | $101.01 | −22.5% |

Extending training data back to 2000 (dot-com crash, 2008 crisis) made the agent trade more actively on volatile stocks (SOFI wins on test and forward), but it still did not beat buy-and-hold overall. Most of the gap came from exiting INTC early during a ~3× rally in 2026.

## Diagnostics: why buy-and-hold?

**1. Reward diagnostic.** Fixed policies were run for 300 episodes in the training environment to see what the reward actually favors:

| Loss penalty | Always cash | Always hold | Random |
| --- | --- | --- | --- |
| 5.0 | −4.12 | −9.12 | −24.93 |
| 1.0 | −4.12 | −1.90 | −20.19 |
| 0.0 | −4.12 | −0.10 | −19.00 |

At 5.0 the reward scored cash above holding, which trapped 4 of 5 seeds in cash. At 0.0, holding scored exactly −0.10 every episode (just the fee) — a "safe zero" trap. 1.0 was chosen as the balance.

**2. Rule-based benchmark.** A simple "hold above the 60-day MA, sell below it" rule lost badly to buy-and-hold on validation ($117.68 vs $151.31) because of whipsaw (15–42 trades per stock). It only helped in long, deep declines.

**3. Fair version comparison.** An earlier version (v5) looked better, but only on a test period that had been used during tuning. On the unseen forward period it was worse than v6.

## What I learned

The balance between reward and punishment was the fundamental factor to consider when building an agent. High punishment results in the agent pausing, as it avoids risking punishment rather than earning reward. Also, variance in AI tells me that AI agents have their own tendency and style of following code even when the code is exact same. However, even after increasing the amount of data, buy-and-hold still performed better. This showed me that the problem was the type of information, not the amount of data.

## Limitations

- 0.1% fee is an assumption; real Webull Canada fees and CAD→USD conversion costs need checking.
- Trades execute at the same day's close; real orders would fill at the next open.
- All-in / all-out positions only.
- The stock universe has survivorship bias (all companies still exist today).

## Next: Phase 2

Add **new information** instead of new price features: news headline sentiment scored with [FinBERT](https://huggingface.co/ProsusAI/finbert), aligned by timestamp to prevent lookahead bias. First test whether sentiment predicts returns at all, then add it to the agent's observations.

## Tech stack

Python · Google Colab · yfinance · pandas · NumPy · Gymnasium · Stable-Baselines3 (PPO)

## How to run

1. Open `RL_Swing_Trader_v6.ipynb` in Google Colab.
2. Run cells 1–5 (Drive mount, installs, data download, features, environment).
3. Cell 7 trains and evaluates on validation; cell 8 runs the final test once.

## About

Built by David Jeon, first-year Engineering student at the University of Alberta, with AI pair-programming assistance (Gemini, Claude) for code and debugging.
