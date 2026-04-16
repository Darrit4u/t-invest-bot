# Server Paper Runtime

Этот документ описывает запуск и эксплуатацию непрерывного paper-trading runtime.

## Назначение

`run_server_paper.py` запускает долгоживущий процесс, который:

- получает рыночные данные (`demo` или `t_invest`)
- прогоняет полный pipeline стратегий/фильтра/портфеля/исполнения
- ведет paper-позиции и сделки
- отправляет события в Telegram
- шлет heartbeat и daily report
- восстанавливает открытые позиции после рестарта
- защищен от дублей по обработке баров/сигналов/уведомлений

## Запуск

```powershell
python run_server_paper.py --config-dir .\config --log-dir .\logs
```

Проверка конфигов без старта:

```powershell
python run_server_paper.py --check-config
```

Тестовый автостоп:

```powershell
python run_server_paper.py --run-seconds 600
```

## Ключевые настройки (`params.yaml`)

```yaml
runtime:
  mode: server_paper
  polling_interval_sec: 30
  heartbeat_enabled: true
  heartbeat_interval_min: 180
  daily_report_enabled: true
  daily_report_time: "23:10"
  timezone: "Europe/Moscow"
  dedup_enabled: true
  restart_recovery_enabled: true
```

Telegram-флаги:

```yaml
telegram:
  enabled: true
  send_signals: true
  send_positions: true
  send_daily_report: true
  send_heartbeat: true
  send_errors: true
```

## Идемпотентность и рестарт

Для надежности runtime сохраняет:

- `last_processed_by_stream` (последний обработанный бар по потоку)
- список отправленных daily reports
- журнал отправленных runtime-уведомлений
- snapshot trade-state в `trades` и lifecycle в `trade_events`

При рестарте:

- открытые сделки поднимаются из SQLite и восстанавливаются в симуляторе
- уже обработанные бары не запускают pipeline повторно
- daily report повторно за тот же день не отправляется
- дубли сигналов фильтруются по origin (`instrument + strategy + timestamp`) и persisted state

## Ограничения

- Это paper runtime, реальные ордера брокеру не отправляются.
- Daily report использует `equity_change_proxy = realized + unrealized`.
- Weekly summary в текущей реализации не включен (можно добавить следующим этапом).
