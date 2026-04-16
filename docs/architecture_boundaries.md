# Architecture Boundaries (Stabilization)

This note defines which contracts are official for upper layers and which are internal runtime details.

## Official domain contract (upper layers)

Upper layers (`signal processor`, `portfolio`, `stats`, `notifier`, reporting) should operate on:

- `Signal`
- `Position`
- `Trade`
- `DomainEvent` (`SignalAccepted`, `SignalRejected`, `PositionOpened`, `PositionUpdated`, `PositionClosed`, `TradeClosed`, `RiskRejected`, `AllocationRejected`)

`core/portfolio_events.normalize_trade_event(...)` is the normalization boundary from simulator events to domain events.

## Low-level simulator contract (internal)

`TradeEvent` is a low-level lifecycle event emitted by `TradeSimulator`.

Allowed usage:

- inside simulator/execution internals
- audit/debug persistence
- normalization adapter (`TradeEvent -> DomainEvent`)

Not allowed as a primary contract for upper layers.

## Storage boundary

`SQLiteStore` keeps:

- `signals`: accepted signal snapshots
- `trades`: simulator trade-state snapshots (technical state for runtime/audit)
- `trade_events`: low-level simulator lifecycle events (technical audit trail)
- `stats_snapshots`: periodic summary snapshots

Storage rows are persistence details, not domain objects.
Upper layers should consume domain entities/events via adapters, not raw SQLite rows.

## Compatibility policy

Backward-compatible aliases remain in place for now:

- `save_trade(...)` -> alias to `save_trade_state_snapshot(...)`
- `save_trade_event(...)` -> alias to `save_trade_lifecycle_event(...)`

This preserves existing flows while keeping storage intent explicit.
