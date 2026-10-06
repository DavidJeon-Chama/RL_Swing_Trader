# RL-Swing-Trader — Phase 1 Final Report

**Author:** David Jeon · **Updated:** October 5, 2026

## Summary

A PPO agent trained on 14 U.S. stocks under realistic $100 cash-account rules converged to buy-and-hold. On unseen 2026 data it averaged **$109.36 per $100** across 5 seeds, versus **$113.20** for buy-and-hold, and diagnostic experiments explained why.

- Built a custom Gymnasium trading environment: T+1 settlement, 0.1% fee, 3-day minimum hold, benchmark-relative reward.
- Found reward-design flaws with fixed-policy diagnostics: a loss penalty made staying in cash score higher than holding (−4.12 vs −9.12 per episode), which trapped 4 of 5 seeds in cash.
- Used honest evaluation: date-based train / validation / test / forward splits, a held-out ticker (KO), and multiple random seeds.
- A hand-made 60-day moving-average rule also lost to buy-and-hold ($101.01 vs $113.20) and cut drawdown by only 3.2 points.
- Extending training data back to 2000 (dot-com crash, 2008 crisis) made the agent trade more actively, but it still did not beat buy-and-hold.
- **Conclusion:** price-only signals showed no edge after costs. Phase 2 will add new information through news sentiment (FinBERT).

## 1. Project Overview and Environment Design

The goal was a reinforcement learning bot that swing-trades in a Webull cash account ($100) while covering trading costs. Phase 1 used price data only, built with Python, Google Colab, yfinance, Gymnasium, and Stable-Baselines3 (PPO).

| Component | Final design (v6) |
| --- | --- |
| Actions | 0 = wait / keep position, 1 = buy (all-in), 2 = sell (all-out) |
| Account observations (5) | cash ratio, unsettled cash ratio, in-position flag, unrealized P&L, holding progress |
| Market observations (5) | 5-day momentum, distance from 20-day MA, distance from 60-day MA, drawdown from 60-day high, 20-day volatility |
| Realistic constraints | T+1 settlement, 0.1% fee per trade, 3-day minimum holding period |
| Reward | excess return vs buy-and-hold × 100 + down-day loss penalty + losing-streak penalty on losing sells |
| Data | 14 U.S. stocks (AAPL, SOFI, F, PEP, MSFT, JNJ, XOM, T, DIS, NKE, WMT, BA, GE, INTC) + KO as a validation-only holdout |
| Splits | Train up to 2022 · Validation 2023–2024 · Test 2025 · Forward Jan–Sep 2026 |

Each training episode is a random 250-trading-day window from one of the 14 stocks. The minimum holding period blocks high-frequency trading by rule rather than by penalty.

## 2. Development Timeline (newest first)

Each version fixed the reason the previous results looked wrong. The most valuable habit was checking every "why?" with numbers.

| Version | What changed | Problem found |
| --- | --- | --- |
| v6 + 2000 data | Training data extended to 2000; 3 seeds × 500k steps | Agent traded more on volatile stocks, but still did not beat buy-and-hold |
| v6 (loss penalty 1.0) | Bug fixes, start-independent features, unrealized P&L, date splits | Escaped the cash trap, but almost every seed copied buy-and-hold |
| v6 (first attempt) | Same fixes with loss penalty 5.0 | 4 of 5 seeds never bought — the reward scored cash higher than holding |
| v5 | 20-day moving average, losing-streak penalty | INTC loss seemed solved ($31.84 → $260.39), but on a period used for tuning, so scores were inflated. Streak penalty was applied every step (bug) |
| v4 | 14 stocks, 5 seeds, 300k steps | Seed 100 traded INTC 14 times, repeatedly buying during a decline ("catching a falling knife") |
| v3 | Random training across 4 stocks, loss penalty | Collapsed to zero trades on KO, a stock never used in training — overfitting |
| v2 | Price and momentum observations, benchmark-relative reward | The observations had contained no price at all. All 3 seeds converged to buy-and-hold |
| v1 | T+2 settlement, minimum holding period, return-based reward | Results split to extremes by seed: $100 (cash) or $142 (buy-and-hold) |

## 3. Diagnostic Experiments and Results

All four experiments pointed to the same conclusion: price indicators alone could not find a strategy that beats buy-and-hold after fees.

### 3.1 Reward diagnostic

Fixed policies (no AI) were run for 300 episodes each in the training environment, comparing the average total reward. If cash scores higher than holding, the agent has no choice but to learn to stay in cash.

| Loss penalty | Always cash | Always hold | Random |
| --- | --- | --- | --- |
| 5.0 | −4.12 | −9.12 | −24.93 |
| 1.0 | −4.12 | −1.90 | −20.19 |
| 0.0 | −4.12 | −0.10 | −19.00 |

At 5.0, cash was the better choice, so 4 seeds got stuck in cash. At 0.0, holding scored exactly −0.10 (the fee) every episode, creating a "just hold and stay safe" trap. 1.0 was chosen as the balance. The same test on 2000+ data gave the same ordering (cash −6.15, hold −1.91 at 1.0).

### 3.2 Fair comparison on unseen data

Average results across seeds on the forward period (Jan–Sep 2026), which no model had seen, across 15 stocks:

| Strategy | Avg final value ($100 start) | Avg max drawdown |
| --- | --- | --- |
| Buy-and-hold | $113.20 | −25.7% |
| v6 (5 seeds) | $109.36 | about −25% |
| v5 (5 seeds) | $106.79 | 0% to −25.7% by seed |
| 60-day MA rule | $101.01 | −22.5% |

v5 had looked better earlier only because it was scored on a test period used during tuning. v5 seed 100 kept drawdown to −4.7%, but it mostly stayed in cash, so it earned only $99.71.

### 3.3 Rule-based benchmark

A "hold above the 60-day MA, sell below it" rule lost badly on validation: $117.68 vs $151.31 for buy-and-hold. It traded 15–42 times per stock, buying high and selling low (whipsaw). It only helped in long, deep declines, such as INTC on validation (drawdown −62% → −35%).

### 3.4 More data: training from 2000

Training data was extended back to 2000 to include the dot-com crash and the 2008 financial crisis (3 seeds × 500k steps; SOFI keeps its shorter history).

| Seed | Validation avg | Win rate vs buy-and-hold | Avg trades |
| --- | --- | --- | --- |
| 0 | $138.76 | 7% | 1.9 |
| 42 | $135.51 | 7% | 7.4 |
| 100 | $147.60 | 29% | 4.4 |
| Buy-and-hold | $151.31 | — | — |

The agent traded more than before and won more often on volatile stocks (SOFI won on both test and forward). On the forward period, the best seed averaged $102.75 vs $112.14 for buy-and-hold on the same 14 stocks. Almost all of that gap came from one stock: INTC rose about 3× in 2026, and the agent exited early ($169 vs $294). The behavior changed, but the conclusion did not.

## 4. Conclusion: Why Buy-and-Hold?

In this setup (five price indicators, 0.1% fee, 2000–2026 data), buy-and-hold was effectively the best strategy, and the agent found it correctly. It did not converge there because it was "lazy."

- Every timing attempt trades a certain cost (0.2% round-trip fee) for an uncertain gain. The price indicators did not carry enough information to reduce that uncertainty.
- The 60-day MA rule, a direct version of "buy when it goes up, sell when it goes down," also lost because of whipsaw. Price trends alone cannot tell a short dip from a real decline.
- Adding 23 years of data, including long bear markets, changed how the agent behaved but not whether it could beat buy-and-hold.

The problem was the information, not the algorithm. Showing the chart as an image would give the same price information in a different shape. The agent needs new information that is not already reflected in the price.

## 5. How to Read the Result Tables

Every result starts with $100 per stock. Each row means "if the bot had traded this stock during this period."

| Column | Meaning | Example |
| --- | --- | --- |
| ticker | Stock symbol | INTC = Intel |
| bot | Bot's final value at the end of the period | 101.71 → $100 became $101.71 (+1.7%) |
| buy_hold | Final value if bought on day one and held | 121.88 → +21.9% |
| beat | Did the bot earn more than buy-and-hold? | False = lost |
| trades | Number of buys plus sells | 0 = never bought, 1 = bought and held, 2 = one round trip |
| max_dd_% | Maximum drawdown: the largest fall from a peak to a later low | −12.8 → at one point fell 12.8% below its peak |
| bh_mdd_% | Buy-and-hold's maximum drawdown | Shows how much risk the bot reduced |
| win rate / beat_% | Share of stocks where the bot won | 60% = won 9 of 15 |

Two cautions. Win rate counts how many times the bot won, not by how much: winning 9 times by a little and losing 6 times by a lot gives a high win rate but a lower average. In a falling market, staying in cash ($100) also counts as a "win," so always read the average value and drawdown together.

**One-line explanation for others:** "The table compares what $100 became with the bot (bot), what it would have become if you just bought and held (buy_hold), and the worst drop along the way (MDD)."

## 6. Limitations and Phase 2 Plan

### Known limitations

- The 0.1% fee is an assumption. Real Webull Canada fees for U.S. stocks and CAD→USD conversion costs still need to be checked.
- Fractional share purchases are assumed; availability in the real account needs to be confirmed.
- Orders fill at the same day's close the agent observed. Real orders would fill at the next open, so results are slightly optimistic.
- Positions are all-in / all-out. With a $100 account, splitting into smaller positions has little practical benefit, so this was accepted.
- The 14 stocks are well-known companies that still exist today, which introduces survivorship bias. This was accepted for a personal project.

### Phase 2: adding new information

- [ ] Score news headlines with FinBERT and add sentiment to the agent's observations
- [ ] First test, without RL, whether sentiment predicts the next days' returns at all
- [ ] Timestamp all news and align it to trading days to block lookahead bias (seeing news before it was published)
- [ ] Evaluate social media and policy announcements (X, Truth Social) as sources: X's API is pay-per-use with expensive historical search, and Truth Social's official data feed targets enterprises, so start with free historical news datasets and collect live data going forward
