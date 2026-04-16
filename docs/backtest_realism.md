# Backtest Realism Assumptions

This project uses a deliberately conservative, reproducible simulation model.

## Execution timing

- Signals are formed on a closed bar.
- Activation is allowed only from the **next** bar (`candle.datetime > signal.timestamp`).
- This prevents same-bar lookahead fills.

## MTF aggregation

- Higher-timeframe candles are built from lower-timeframe bars.
- Only fully formed higher-timeframe buckets are used by MTF filters.
- Incomplete buckets are ignored to avoid lookahead bias.

## Gap and intrabar behavior

- Intrabar stop/take conflict uses pessimistic ordering when configured (`intrabar_stop_priority`).
- Gap handling can use conservative open-price logic via lifecycle policy.

## Costs

- Commission is charged per side on each fill (entry, partial exits, final exit).
- Optional slippage model:
  - `execution.slippage.model: fixed_ticks`
  - independent ticks for `entry`, `stop_exit`, `target_exit`, `forced_exit`
  - optional per-instrument overrides.

## Liquidity and futures sanity checks

- Optional minimum bar liquidity filter: `signal_filter.min_bar_volume_ratio`.
- Optional near-expiry block: `futures.block_near_expiry`, `futures.expiry_buffer_days`, `futures.expiries`.
- Prices used for fills are normalized to instrument tick size.

## Known simplifications (kept intentionally)

- No order-book simulation.
- No latency queue simulation.
- No exchange-grade contract roll engine yet.
- Slippage is rule-based, not microstructure-based.
