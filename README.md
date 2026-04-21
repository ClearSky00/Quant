# AlgoTrader

A high-frequency multi-strategy algorithmic trading system built on Alpaca Markets. Runs a fully event-driven WebSocket architecture for live/paper trading and a bar-by-bar simulation engine for backtesting.

## Strategies

Four independent intraday strategy layers run concurrently on a universe of large-cap equities:

| Layer | Strategy | Description |
|-------|----------|-------------|
| L1 | VWAP Mean Reversion | Fades VWAP deviations with RSI confirmation |
| L2 | Opening Range Breakout | Trades breakouts from the 09:30–10:00 opening range |
| L3 | RSI Reversal Scalp | Short-duration oversold bounces on 7-period RSI |
| L4 | Volume Surge Momentum | Enters on abnormal volume with strong bar positioning |

A market regime filter (SPY 200-SMA + VIXY proxy) scales position sizes and halts trading in high-volatility environments.

## Architecture

```
WebSocket bars → BarStore → BarDispatcher → Strategy Layers → OrderManager → Alpaca
                                                            ↘ RiskManager (PDT, circuit breaker, stop loss)
                                                            ↘ NotificationQueue → Discord
```

- **StreamEngine** — async WebSocket consumer, routes 1-min bars to all strategy layers
- **BarDispatcher** — fan-out coordinator, enforces max positions and layer arbitration
- **APScheduler** — pre-market warmup (08:00), market open reset (09:30), ORB finalize (10:00), EOD close (15:30), daily report (16:30)
- **BacktestEngine** — bar-by-bar historical replay with slippage modeling, equity curve, and trade log output

## Risk Management

- Per-trade dollar risk cap ($150) with ATR-based stop placement
- Max daily loss ($1,500) and drawdown ($5,000) circuit breakers
- PDT guard — limits day trades to 3 per week
- Regime-based position size scaling (50% reduction above VIXY threshold, full halt above hard threshold)
- Trailing stop activation at 1.0R, trailing at 40% of move
- EOD forced close at 15:30 ET

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # add ALPACA_API_KEY, ALPACA_SECRET_KEY, DISCORD_WEBHOOK_URL
```

```bash
# Paper trading (default)
python main.py --paper

# Dry run — full signal generation, no real orders
python main.py --dry-run

# Historical backtest (365 days)
python main.py --test

# Backtest from a specific date
python main.py --test --start 2024-01-01

# Generate dashboard report only
python main.py --report-only
```

## Configuration

All parameters are in `config.yml` — strategy thresholds, risk limits, scheduler times, and universe filters. No code changes needed to tune behavior.

Key defaults:
- Starting capital: $100,000
- Max concurrent positions: 15
- Universe: large-cap equities (price ≥ $15, ADV ≥ $50M)
- Target trade cadence: ≥300 trades/year

## Stack

- **Python 3.11+** — asyncio, APScheduler
- **Alpaca Markets** — brokerage API + WebSocket data feed
- **Supabase / SQLite** — trade and portfolio state persistence
- **Discord** — real-time trade alerts and daily summaries
- **Docker** — `docker-compose up` for containerized deployment
