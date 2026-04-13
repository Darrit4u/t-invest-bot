# No-Trade Analysis CLI

Скрипт `scripts/find_no_trade_windows.py` анализирует отчёты matrix backtest и предлагает `no-trade` окна для фильтрации входов по паре `instrument + strategy`.

## Что делает

- Читает `trades.csv` (и использует только закрытые сделки `closed_at != ""`).
- Берёт время сделки из `created_at` или `activated_at` (через `--time-source`).
- Конвертирует время в локальную таймзону инструмента (по `config/instruments.yaml` + `session_rules`).
- Считает метрики по бакетам:
  - `hour_local`,
  - `weekday_local`,
  - опционально `hour_local + weekday_local`.
- Делит сделки по времени на `train/validation` и подтверждает окно только если оно плохое в обеих частях.
- Формирует артефакты:
  - `no_trade_summary.md`,
  - `no_trade_windows.json`,
  - `no_trade_patch.yaml`.

## Пример запуска

```powershell
python scripts/find_no_trade_windows.py `
  --reports-dir backtest_reports `
  --report latest `
  --min-trades-per-bucket 10 `
  --min-winrate-gap-pp 12 `
  --min-negative-avg-r 0 `
  --validation-ratio 0.3 `
  --output-dir backtest_reports `
  --time-source created_at `
  --profile active
```

## Полезные параметры

- `--instrument SILVER` — анализ только одного инструмента.
- `--strategy trend_pullback_vwap_ema` — анализ только одной стратегии.
- `--include-combined` — добавить анализ бакетов `weekday+hour`.
- `--min-trades-per-day 1.0` — не предлагать фильтры, если ожидаемая частота упадёт ниже этого уровня.

## Пример `no_trade_patch.yaml`

```yaml
strategy_params:
  by_instrument:
    SILVER:
      trend_pullback_vwap_ema:
        blocked_entry_hours_local: [10, 11]
        blocked_entry_weekdays_local: [0]
```

Примечание: точное содержимое patch зависит от текущего отчёта и выбранных порогов CLI.
