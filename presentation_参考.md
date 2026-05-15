# Group Presentation Script — QuantumTrade v2

**25 min + Q&A | Fintech BA3098 | Dr. Peimin Chen**

---

## Timing Overview

| # | Segment | Time | Speaker |
|---|---------|------|---------|
| 1 | Opening + Problem Definition | 2-3 min | |
| 2 | System Architecture Overview | 2 min | |
| 3 | NLP Pipeline | 4-5 min | |
| 4 | RL Trading Engine | 5 min | |
| 5 | Ablation Results | 4-5 min | |
| 6 | Dashboard Live Demo | 3-4 min | |
| 7 | Critical Reflection | 2 min | |
| — | **Total** | **22-26 min** | |

**Buffer**: If running over, cut the NLP method comparison details (VADER vs LR). If running under, expand the ablation sector analysis.

---

## Segment 1: Opening + Problem Definition (2-3 min)

### Slide: Title Slide
- Project name: NLP-Driven Reinforcement Learning Trading Platform
- Team members
- Fintech BA3098, Dr. Peimin Chen

### Speaking Script

> Good morning/afternoon. Our project is an NLP-driven reinforcement learning trading platform. The core question we set out to answer is simple to state but difficult to test rigorously:
>
> **Do NLP-derived sentiment signals improve RL trading performance?**
>
> Let me make this concrete. Every day, thousands of financial news articles are published — earnings reports, product launches, regulatory decisions, CEO interviews. Human traders read these and adjust their positions. But can a machine do the same thing?
>
> Traditional quantitative trading uses only price data — OHLCV bars, technical indicators. It completely ignores the vast universe of textual information. Our platform bridges this gap. We built a system that:
> 1. Fetches real financial news
> 2. Analyzes sentiment using three NLP methods, including FinBERT
> 3. Feeds that sentiment signal into a Double DQN trading agent
> 4. And — crucially — runs a controlled ablation experiment to measure whether NLP actually helps
>
> We tested this across 28 S&P stocks, training 168 DQN models with multi-seed validation. I'll walk you through what we built, how we tested it, and what we found.

### Slide: Research Question Visual
- Big text: "Do NLP signals improve RL trading?"
- Simple diagram: Raw News → Sentiment Score → DQN State → Buy/Sell/Hold
- Bottom: 28 stocks × 2 conditions × 3 seeds = 168 models

---

## Segment 2: System Architecture Overview (2 min)

### Slide: 5-Module Pipeline Diagram

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  ① Data      │    │  ② NLP       │    │  ③ Data      │    │  ④ RL Engine │    │  ⑤ Dashboard │
│  Ingestion   │───▶│  Pipeline    │───▶│  Storage     │───▶│  (DQN)       │───▶│  (Streamlit) │
│  Yahoo/      │    │  VADER/LR/   │    │  SQLite      │    │  Gym Env +   │    │  8 panels    │
│  Finnhub     │    │  FinBERT     │    │  6 tables    │    │  Double DQN  │    │  + RAG Chat  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                                       │
       └───────────────────┴───────────────────────────────────────┘
                        Data flows left to right.
              Sentiment score from Module ② feeds into Module ④ state vector.
```

### Speaking Script

> Our architecture has five modules in a pipeline. Data flows left to right.
>
> Module 1 fetches market data and news from multiple APIs with a fallback chain — if Yahoo Finance rate-limits us, we fall back to Alpha Vantage, then to synthetic data.
>
> Module 2 is the NLP pipeline — three sentiment methods: VADER with a custom financial lexicon, TF-IDF Logistic Regression, and FinBERT. The output is a scalar sentiment score in [-1, +1].
>
> Module 3 persists everything in SQLite — six normalized tables.
>
> Module 4 is the RL engine — a custom Gym environment with an 11-dimensional state vector and a Double DQN agent built from scratch. This is where the sentiment score enters the state.
>
> Module 5 is the Streamlit dashboard — real-time visualization of the entire pipeline.
>
> The core integration requirement — sentiment from Module 2 in the RL state vector of Module 4 — is satisfied at the boundary between these two modules.

---

## Segment 3: NLP Pipeline (4-5 min)

### Slide: Three Sentiment Methods Comparison

| Method | Type | Speed | Accuracy | Our Enhancement |
|--------|------|-------|----------|-----------------|
| VADER | Lexicon-based | ~1ms/article | Baseline | +14 financial domain terms |
| Logistic Regression | Traditional ML | ~5ms/article | Moderate | TF-IDF + cold-start labels |
| FinBERT | Transformer DL | ~50ms/article | Best | ProsusAI/finbert, domain pre-trained |

### Slide: Preprocessing Pipeline
Lowercase → Remove URLs/HTML → Remove punctuation → Tokenize → Remove stopwords → Lemmatize

### Slide: Sentiment Quad Chart (screenshot from Dashboard)
Show the 4-method overlay for a well-known stock (AAPL or TSLA). Point out:
- Where all 3 methods agree (strong signal)
- Where they diverge (ambiguous news)
- FinBERT (blue) typically shows more nuanced scores than VADER (green)

### Speaking Script

> The NLP pipeline transforms raw news text into a single number: a sentiment score between -1 and +1. We use three complementary methods.
>
> **VADER** is a rule-based lexicon approach. It's extremely fast — under 1 millisecond per article — but it struggles with financial jargon. We augmented its dictionary with 14 domain-specific terms. For example, "bullish" gets a +1.8 weight in our financial lexicon.
>
> **Logistic Regression** is our traditional ML baseline. We vectorize text with TF-IDF and train a classifier. Since labeled financial news isn't available upfront, we use keyword heuristics to generate cold-start training labels.
>
> **FinBERT** is our primary method. It's a BERT-base model fine-tuned on SEC filings and financial reports. It understands that "profit decreased 20%" is negative even if the CEO says "outlook remains positive" — because it's learned that financial data carries more weight than management rhetoric.
>
> We compute a confidence-weighted ensemble score across all three methods. We also measure inter-rater agreement with Fleiss' Kappa and compute F1 scores using FinBERT as pseudo-ground-truth. This gives us both a best-guess sentiment signal and a measure of how reliable that signal is.
>
> [Show sentiment quad chart]
>
> This is what the output looks like in our dashboard. Green is VADER, yellow is LR, blue is FinBERT. When all three lines move in the same direction, the sentiment signal is strong. When they diverge — that's informative too: it tells us the news is ambiguous.

---

## Segment 4: RL Trading Engine (5 min)

### Slide: State Vector Design

| Dim | Feature | Formula | What it tells the agent |
|-----|---------|---------|------------------------|
| 0 | price_ratio | close / initial_close | Where are we relative to start? |
| 1 | MA50_ratio | MA50 / close | Short-term trend |
| 2 | MA200_ratio | MA200 / close | Long-term trend |
| 3 | RSI_norm | RSI(14) / 100 | Overbought or oversold? |
| 4 | MACD_ratio | MACD / close | Trend strength |
| 5 | position_pct | position / portfolio | How much are we invested? |
| 6 | cash_pct | cash / portfolio | How much dry powder? |
| 7 | **sentiment_score** | NLP ensemble | **Raw sentiment signal** |
| 8 | **sentiment_ma5** | 5-day EMA | **Smoothed sentiment** |
| 9 | **sentiment_trend** | MA5 - MA20 | **Sentiment momentum** |
| 10 | **sentiment_vol** | 10-day std | **Sentiment uncertainty** |

Highlight: dimensions 7-10 (bold) are the NLP contribution. The ablation removes these 4 dimensions.

### Slide: DQN Architecture
```
Input(11) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Output(3)
                                                                         Hold | Buy | Sell
```
- ~44,000 parameters
- Double DQN: online network selects action, target network evaluates it
- Target network hard-copied every 5 episodes

### Slide: Reward Function
```
R = ΔPortfolio% − 0.0001 × idle_cash − 0.0005 × excess_concentration
    − 0.0001 × trade_penalty ± 0.0002 × sentiment_alignment_bonus
```
- Primary signal: portfolio value change
- Three risk penalties: cash idling, position concentration, overtrading
- One shaping reward: position-sentiment alignment

### Slide: Training Configuration
- 200 episodes, ε-greedy: 1.0 → 0.05 (decay 0.97)
- Experience replay: 20,000 capacity, batch size 64
- Walk-forward: 60% train / 20% val / 20% test (chronological split)
- Multi-seed: 3 seeds [42, 123, 456] per condition

### Slide: Convergence Plot (screenshot)
Show training reward curve — mean episode reward over 200 episodes, with shaded std across 3 seeds.

### Speaking Script

> Now the core of the platform: the RL trading engine.
>
> Every trading day, the agent sees an 11-dimensional state vector. Seven dimensions come from price-based technical indicators — things like RSI, MACD, moving averages. Four dimensions come from our NLP pipeline — the raw sentiment score, its 5-day smoothed average, its trend, and its volatility.
>
> The agent has three actions: Hold, Buy 25% of available cash, or Sell 25% of current position. We use 25% trade fractions to prevent all-in behavior — this is standard institutional practice.
>
> The reward function has five components. The primary driver is portfolio return. On top of that, we layer four risk controls: a penalty for sitting in cash, a penalty for excessive concentration in one stock, a penalty for overtrading, and a small sentiment alignment bonus that rewards the agent when its position matches the NLP signal direction.
>
> The network architecture is a standard funnel: 11 inputs, three hidden layers of 256, 128, and 64 neurons, and 3 outputs — the Q-values for Hold, Buy, and Sell. We use Double DQN specifically because it reduces overestimation bias. The online network picks the action, the target network evaluates it, and they sync every 5 episodes.
>
> Training runs for 200 episodes with epsilon-greedy exploration decaying from 1.0 to 0.05. Experience replay with a 20,000-transition buffer breaks temporal correlation. Gradient clipping at norm 1.0 prevents exploding gradients.
>
> Crucially, all validation is walk-forward — 60% train, 20% validation, 20% test, split chronologically. The model never sees future data during training. The best checkpoint is selected by validation Sharpe, not terminal training performance.
>
> [Show convergence plot]
>
> This shows the learning curve for one stock. The shaded region is standard deviation across three seeds. You can see the agent transitions from random exploration — negative rewards — to positive returns as epsilon decays and it exploits what it's learned. But note the variance — this is a key finding I'll return to.

---

## Segment 5: Ablation Results (4-5 min)

### Slide: Ablation Study Design
```
For each of 28 tickers:
  With-NLP (11-dim state)  ←→  Without-NLP (8-dim state)
         ↓                              ↓
    Train DQN                      Train DQN
         ↓                              ↓
    Backtest                       Backtest
         ↓                              ↓
    ΔSharpe = Sharpe(with) − Sharpe(without)
```
× 3 seeds per condition = 168 total training runs

### Slide: Layer 1 — NLP Contribution

| Category | Count | % |
|----------|-------|---|
| NLP Improves Sharpe | 12 / 28 | 42.9% |
| NLP Neutral | 5 / 28 | 17.9% |
| NLP Reduces Sharpe | 11 / 28 | 39.3% |

Top performers: DIS (+1.978), MSFT (+1.602), ABBV (+0.855)
Worst performers: NVDA (−0.861), NKE (−0.811), V (−0.744)

### Slide: Layer 2 — DQN vs Buy-and-Hold

| Sector | Avg DQN Sharpe | Avg BH Sharpe | DQN Wins |
|--------|---------------|---------------|----------|
| Tech | −0.717 | +0.715 | 3/7 |
| Finance | −0.194 | −0.121 | 3/5 |
| Healthcare | +0.599 | +1.379 | 2/4 |
| Consumer | −0.344 | −0.625 | 3/5 |
| Energy/Industrial | +1.279 | +1.507 | 2/5 |
| Comm/Utility | +2.600 | +2.237 | 1/2 |

**Overall Sharpe win rate: 50.0% (14/28)**

### Slide: Layer 3 — Paper Trading
- 21 BUY, 7 SELL
- Portfolio: $99,900.84 (−0.10%)
- Breakeven result, costs equal to transaction fees

### Slide: Key Finding Visual
Two-column comparison:
- **Left**: DIS (NLP helps) — rich media ecosystem, event-driven news, NLP ΔSharpe = +1.978
- **Right**: NVDA (NLP hurts) — saturated news coverage, repetitive sentiment, NLP ΔSharpe = −0.861

### Speaking Script

> Now the results — the answer to our core question.
>
> [Show ablation design]
>
> For each of 28 stocks, we train two DQN agents: one with the full 11-dimensional state including NLP sentiment, and one with only 8 dimensions — the price features alone. Everything else is identical. Three seeds per condition to measure variance. In total, 168 models trained over roughly 8 hours.
>
> [Show Layer 1]
>
> **Layer 1: Does NLP help?** The answer is nuanced. In 12 out of 28 stocks — 42.9% — NLP improves the Sharpe Ratio. The best case is Disney, with a Delta-Sharpe of +1.978. DIS has a rich media ecosystem — box office results, streaming numbers, theme park attendance, CEO changes — all of which generate news that carries information beyond price data.
>
> But in 11 stocks — 39.3% — NLP actually degrades performance. The worst case is NVIDIA, with Delta-Sharpe of −0.861. NVDA has the opposite problem: it's covered so intensely that every news outlet says the same thing. The NLP signal flatlines — it becomes a constant, not a signal. The DQN overfits to it and misses real price movements.
>
> [Show Layer 2]
>
> **Layer 2: Can DQN beat Buy-and-Hold?** On Sharpe Ratio, it's a coin flip — 14 wins, 14 losses, exactly 50%. The sector breakdown is revealing: DQN performs best where Buy-and-Hold also performs well, suggesting it's capturing trend-following behavior rather than generating independent alpha.
>
> [Show Layer 3]
>
> **Layer 3: Paper trading.** We deployed the trained agents on the most recent unseen data. 21 buy signals, 7 sell signals. Portfolio value: $99,900.84 — a loss of exactly 0.1%, which equals our transaction costs. Net of costs, the strategy is breakeven.
>
> [Show key finding]
>
> The headline finding is **context dependence**. NLP is not a universal performance booster. It helps on stocks where news carries orthogonal information — event-driven stocks like DIS and ABBV. It hurts on stocks where news is either too sparse or too saturated. This finding has practical implications: deploying NLP signals selectively, rather than uniformly, would likely improve aggregate performance.

---

## Segment 6: Dashboard Live Demo (3-4 min)

### Pre-flight Checklist (5 min before presentation)
- [ ] `streamlit run dashboard/app.py` running on localhost:8501
- [ ] DB file present, data loaded
- [ ] Browser window open, not showing any personal information
- [ ] Screen resolution set so dashboard is fully visible

### Demo Flow

**30 sec — Ticker Tape + KPI Cards**
> This is our live dashboard. At the top, a scrolling ticker tape shows real-time prices for our watchlist stocks — red for up, green for down in the Chinese convention. The KPI cards below tell us about data quality at a glance.

**45 sec — Candlestick + Technical Indicators** (click AAPL or MSFT)
> Selecting a stock shows the candlestick chart with MA50 and MA200 overlays. Below, RSI and MACD. The triangular markers show where our DQN agent actually placed trades — you can see it doesn't simply buy every dip or sell every peak, but the overall behavior pattern is reasonable.

**45 sec — Sentiment Quad** (scroll to sentiment section)
> This is our most distinctive visualization. Four sentiment methods overlaid — VADER in green, Logistic Regression in yellow, FinBERT in blue. When all three move together, the signal is strong. When they diverge — that's a different kind of signal: market uncertainty.

**45 sec — Portfolio Performance**
> The equity curve in blue shows portfolio value over time. The red shadow below is drawdown — how far we fell from the peak. Notice the drawdown stays well within our 20% risk limit, which is enforced as a hard stop during training.

**30 sec — AI Chat Assistant**
> The sidebar has an AI assistant powered by RAG. Let me ask it a question. [Type: "What is the sentiment for NVDA recently?"] It searches our news database, retrieves relevant articles, and the LLM generates an answer with source context — it's not making things up.

**15 sec — Ablation Summary Panel**
> Finally, the ablation summary shows our key results directly in the dashboard — NLP positive vs. negative counts, top performers, all pulled live from our results file.

---

## Segment 7: Critical Reflection (2 min)

### Slide: Limitations (honest discussion)

| Limitation | Why it matters | Future direction |
|-----------|---------------|------------------|
| **DQN training instability** | Seed std up to 3.031 — larger than NLP effect size | PPO/SAC, 10+ seeds, ensemble methods |
| **News sparsity** | 5-20% of trading days have news | Earnings calls, SEC 8-K, social media |
| **State vector** | 4/11 dims = sentiment (36%) — potential overweight | Sector-relative strength, VIX, volume profile |
| **Reward engineering** | 5 hand-tuned coefficients | Multi-objective RL, direct Sharpe optimization |
| **Transaction costs** | 0.1% optimistic, no spread/slippage | Spread-based costs varying by ticker liquidity |

### Slide: What We Actually Learned

> Three takeaways:
>
> **First, NLP helps — selectively.** The question isn't "does NLP work?" but "when does NLP work?" Our answer: when news carries stock-specific, price-orthogonal information.
>
> **Second, DQN variance is the elephant in the room.** Single-seed results — common in RL-for-trading papers — can be misleading. Multi-seed reporting should be standard practice.
>
> **Third, not beating the market is not a failure.** A 50% Sharpe win rate against Buy-and-Hold for large-cap S&P stocks is consistent with semi-strong market efficiency. Our contribution is the rigorous experimental framework, not a claim of market-beating alpha.

### Ending
> Thank you. We're happy to take questions.
> [Keep dashboard running on screen during Q&A — it invites technical questions about specific components]

---

## Q&A Preparation: Anticipated Questions

| Question | Answer Strategy |
|----------|----------------|
| "Why not use PPO/SAC instead of DQN?" | Course requires DQN from scratch. DQN is the most well-documented RL algorithm. We acknowledge PPO/SAC would likely be more stable — see Limitations. |
| "How do you know FinBERT is actually accurate?" | We compute Fleiss' Kappa across methods and F1 scores vs FinBERT as pseudo-ground-truth. Without human-labeled financial news, absolute accuracy cannot be claimed — this is a known limitation of financial NLP. |
| "Why only US stocks? What about the Chinese market?" | Data availability: Yahoo Direct + Finnhub work best for US equities. Chinese A-shares would require different data sources (Wind, Tushare). NLP pipeline would need Chinese financial models (CFinBERT). These are engineering extensions, not conceptual limitations. |
| "Could this make money in real trading?" | Probably not yet. 50% win rate vs BH, breakeven paper trading. Before live deployment, we'd need: better risk controls, more stable algorithms, realistic cost models including spread and slippage. This is a research platform, not a production trading system. |
| "Why 3 seeds — isn't that too few?" | Yes — acknowledged in Limitations. 10+ seeds would take ~27 hours. 3 is the minimum for variance estimation. We report standard deviations transparently. |
| "How did you split train/val/test temporally?" | 60/20/20 chronological split. Training: ~2020-2022. Validation: ~2023. Test: ~2024. Test set used exactly once. No future information leakage. |
| "What happens if all data sources fail?" | Synthetic data fallback — every data source has a deterministic synthetic fallback with ticker-specific parameters, so the pipeline never breaks entirely. This proved essential when yfinance was globally rate-limited during our experiment. |
