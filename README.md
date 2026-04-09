# T-Invest Bot (Intraday Futures Signal Engine)

Проект для paper-trading (без реальной отправки ордеров): получает свечи, классифицирует рыночный режим, генерирует сигналы по стратегиям, симулирует сделки, считает статистику и отправляет уведомления в Telegram.

## Что реализовано

- Поток свечей в `demo` или `t_invest` режиме.
- 3 стратегии: `trend_pullback_vwap_ema`, `compression_breakout`, `liquidity_sweep_reversal`.
- Классификация режимов: `TREND`, `COMPRESSION`, `BALANCE`, `NEUTRAL`.
- Централизованный фильтр сигналов (сессии, blackout, RR, комиссии).
- Симулятор жизненного цикла сделок: активация, TP1/TP2/SL, экспирация, отмена по news/session.
- Хранение сигналов/сделок/событий/снимков статистики в SQLite.
- Логи приложения и Telegram.
- Автотесты (unit + integration + replay + edge cases).

## Структура проекта

```text
config/
core/
storage/
strategies/
tests/
main.py
```

## Требования

- Python 3.10+
- pip

## Установка и запуск (полная инструкция)

1. Перейдите в папку проекта:

```powershell
cd t-invest-bot
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv venv
.\venv\Scripts\Activate.bat  #cmd.exe
or
source ./venv/bin/activate  # *unix
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. (Опционально, только для `t_invest` режима) установите SDK T-Invest:

```bash
pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
```

5. Создайте `.env` на основе `.env.example` и заполните токены:

```bash
cp .env.example .env
```

6. Скопировать YAML-конфиги в `config/`.

```bash
cp config/example/* config/
```

7. Запустите приложение:

```powershell
python main.py
```

8. Для тестового прогона на ограниченное время:

```powershell
python main.py --run-seconds 60
```

## Режимы работы

- `demo` (по умолчанию): синтетические свечи, токен T-Invest не нужен.
- `t_invest`: реальные свечи через API T-Invest.

Для `t_invest` обязательно:
- задать `INVEST_TOKEN` в `.env`;
- выставить `MARKET_DATA_MODE=t_invest` в `.env` или `config/params.yaml`;
- заполнить `figi` для инструментов в `config/instruments.yaml` (без `figi` инструмент пропускается в live-подписке).

## Переменные окружения (`.env`)

Файлы подхватываются в порядке:
1. `.env`
2. `.env.example` (только если ключ отсутствует в `.env`)

| Переменная | Обязательность | Значение по умолчанию | За что отвечает |
|---|---|---|---|
| `INVEST_TOKEN` | Да для `t_invest`, нет для `demo` | `""` | Токен доступа к T-Invest API для live-потока свечей |
| `TELEGRAM_BOT_TOKEN` | Нужен при `telegram.enabled=true` | `""` | Токен Telegram-бота |
| `TELEGRAM_CHAT_ID` | Нужен при `telegram.enabled=true` | `""` | ID чата/канала для отправки сообщений |
| `DB_PATH` | Нет | из `params.yaml -> storage.db_path` (`signals.db`) | Путь к SQLite файлу БД |
| `MARKET_DATA_MODE` | Нет | из `params.yaml -> market_data.mode` (`demo`) | Принудительный выбор режима `demo`/`t_invest` |
| `CHECK_EVERY_SECONDS` | Нет | `60` в `.env.example` | Сейчас в коде не используется (зарезервировано) |

## CLI-аргументы `main.py`

| Аргумент | По умолчанию | За что отвечает |
|---|---|---|
| `--config-dir` | `./config` | Папка с YAML-конфигами |
| `--log-dir` | `./logs` | Папка для логов |
| `--run-seconds` | `0` | Автоостановка через N секунд (`0` = бесконечный режим) |
| `--print-every` | `10` | Частота технических snapshot-логов (каждые N обновлений свечей) |

## Конфигурация `config/instruments.yaml`

### Глобальные поля

| Ключ | За что отвечает |
|---|---|
| `history_depth` | Глубина хранения свечей в памяти на поток `(instrument, timeframe)` |
| `default_timeframe` | Таймфрейм, с которым работает ingest/signal pipeline |

### `session_rules`

| Ключ | За что отвечает |
|---|---|
| `timezone` | Таймзона сессии |
| `start` | Начало сессии (`HH:MM` или `HH:MM:SS`) |
| `end` | Конец сессии (`HH:MM` или `HH:MM:SS`) |

Поддерживаются overnight-сессии (например, `23:00`-`02:00`).

### `instruments.<SYMBOL>`

| Ключ | За что отвечает |
|---|---|
| `enabled` | Включение/выключение инструмента |
| `ticker` | Биржевой тикер |
| `class_code` | Код класса инструмента |
| `uid` | Идентификатор инструмента (опционально) |
| `figi` | FIGI для live-подписки на свечи |
| `tick_size` | Шаг цены |
| `tick_value` | Стоимость тика |
| `lot` | Размер лота/контракта |
| `sessions` | Список имен сессий из `session_rules`, в которых можно торговать |

## Конфигурация `config/strategies.yaml`

| Ключ | За что отвечает |
|---|---|
| `strategies.<SYMBOL>` | Список разрешенных стратегий для инструмента |

Доступные имена:
- `trend_pullback_vwap_ema`
- `compression_breakout`
- `liquidity_sweep_reversal`

## Конфигурация `config/params.yaml`

### Общие настройки

| Ключ | За что отвечает |
|---|---|
| `timezone` | Базовая таймзона приложения и `news_blackout` |
| `news_blackout_file` | Имя файла blackout-окон внутри `config/` |
| `max_eval_candles` | Максимум последних свечей для оценки сигналов |

### `market_data`

| Ключ | За что отвечает |
|---|---|
| `mode` | `demo` или `t_invest` |
| `reconnect_delay_seconds` | Задержка между попытками переподключения live-потока |
| `candle_interval_seconds` | Интервал генерации demo-свечей |
| `base_prices.<SYMBOL>` | Базовая цена для demo-генератора по инструментам |

### `storage`

| Ключ | За что отвечает |
|---|---|
| `db_path` | Путь к SQLite БД (если `DB_PATH` не задан в `.env`) |

### `indicator_engine`

| Ключ | За что отвечает |
|---|---|
| `ema_fast` | Период быстрой EMA |
| `ema_slow` | Период медленной EMA |
| `atr_period` | Период ATR |
| `volume_period` | Окно средней volume |
| `slope_period` | Окно расчета slope для EMA/VWAP |
| `crossing_lookback` | Окно подсчета crossing (для режимов) |
| `overlap_window` | Окно overlap-метрики |
| `swing_window` | Окно swing high/low |

### `regime_classifier`

| Ключ | За что отвечает |
|---|---|
| `trend_ema_distance_atr` | Мин. дистанция EMA в ATR для `TREND` |
| `trend_vwap_slope_atr` | Мин. slope VWAP в ATR для `TREND` |
| `trend_crossing_max` | Макс. crossings для `TREND` |
| `compression_range_min_atr` | Нижняя граница range/ATR для `COMPRESSION` |
| `compression_range_max_atr` | Верхняя граница range/ATR для `COMPRESSION` |
| `compression_ema_distance_atr` | Макс. дистанция EMA в ATR для `COMPRESSION` |
| `compression_vwap_slope_abs_atr` | Макс. abs(slope VWAP) в ATR для `COMPRESSION` |
| `compression_overlap_min` | Мин. overlap ratio для `COMPRESSION` |
| `balance_crossing_min` | Мин. crossings для `BALANCE` |
| `balance_ema_distance_atr` | Макс. дистанция EMA в ATR для `BALANCE` |
| `balance_vwap_slope_abs_atr` | Макс. abs(slope VWAP) в ATR для `BALANCE` |

### `signal_filter`

| Ключ | За что отвечает |
|---|---|
| `commission_roundtrip` | Комиссия round-trip для проверки окупаемости TP1 |
| `safety_multiplier` | Запас по ожидаемой прибыли относительно комиссии |

### `trade_simulator`

| Ключ | За что отвечает |
|---|---|
| `commission_per_side` | Комиссия на вход/выход |
| `tp1_size` | Доля позиции, закрываемая на TP1 (0..1) |
| `max_wait_bars` | Лимит баров ожидания активации сигнала |
| `max_trade_bars` | Лимит баров жизни активной сделки |
| `move_stop_to_breakeven` | Перенос стопа в безубыток после TP1 |
| `close_active_on_blackout` | Закрывать ли активные сделки при старте blackout |
| `intrabar_stop_priority` | При касании и TP, и SL внутри одной свечи приоритет у SL |

### `telegram`

| Ключ | За что отвечает |
|---|---|
| `enabled` | Включение Telegram-нотификатора |
| `retry_attempts` | Число попыток отправки одного сообщения |
| `retry_delay_seconds` | Базовая задержка между повторами |
| `request_timeout_seconds` | HTTP timeout при отправке сообщения |
| `queue_maxsize` | Размер внутренней очереди сообщений |
| `summary_interval_seconds` | Интервал авто-сводок (`0` = выключено) |
| `send_startup_message` | Отправлять сообщение о старте |
| `send_shutdown_summary` | Отправлять итоговую сводку при остановке |

### `strategy_params.trend_pullback_vwap_ema`

| Ключ | За что отвечает |
|---|---|
| `impulse_bars` | Кол-во свечей для оценки импульса |
| `impulse_atr_mult` | Мин. размер импульса в ATR |
| `min_bullish_bars_in_impulse` | Мин. bullish-свечей для long-импульса |
| `min_bearish_bars_in_impulse` | Мин. bearish-свечей для short-импульса |
| `volume_impulse_mult` | Мин. отношение объема импульса к среднему |
| `min_vwap_extension_atr` | Мин. удаление от VWAP в ATR |
| `max_vwap_extension_atr` | Макс. удаление от VWAP в ATR |
| `pullback_min_atr` | Мин. глубина pullback в ATR |
| `pullback_max_atr` | Макс. глубина pullback в ATR |
| `pullback_location_mode` | Тип зоны pullback (`ANY`, и др.) |
| `stop_buffer_atr` | Буфер SL в ATR |
| `tp1_r` | TP1 в R |
| `tp2_r` | TP2 в R |
| `entry_timing_mode` | Режим входа (`NEXT_BAR_OPEN`, `CONFIRMATION_CLOSE`) |

### `strategy_params.compression_breakout`

| Ключ | За что отвечает |
|---|---|
| `compression_window_bars` | Окно поиска компрессии |
| `range_max_atr` | Макс. ширина диапазона в ATR |
| `range_min_atr` | Мин. ширина диапазона в ATR |
| `ema_distance_max_atr` | Макс. дистанция EMA в ATR |
| `vwap_slope_abs_max_atr` | Макс. abs slope VWAP в ATR |
| `overlap_ratio_min` | Мин. overlap ratio |
| `volume_floor_mult` | Мин. baseline объема |
| `breakout_body_min_atr` | Мин. размер тела breakout-свечи в ATR |
| `breakout_volume_mult` | Мин. volume для подтверждения breakout |
| `late_breakout_extension_atr` | Макс. «опоздавшее» расширение в ATR |
| `large_breakout_retest_threshold_atr` | Порог «слишком большого» breakout для retest |
| `stop_atr` | Компонента стопа от ATR |
| `stop_range_factor` | Компонента стопа от ширины диапазона |
| `tp1_r` | TP1 в R |
| `tp2_r` | TP2 в R |
| `max_retest_bars` | Лимит баров на ретест |
| `retest_tolerance_atr` | Допуск ретеста в ATR |

### `strategy_params.liquidity_sweep_reversal`

| Ключ | За что отвечает |
|---|---|
| `reference_lookback_bars` | Окно поиска reference-уровней |
| `balance_crosses_vwap_min` | Мин. пересечений VWAP для balance-фильтра |
| `ema_distance_max_atr` | Макс. дистанция EMA в ATR |
| `vwap_slope_abs_max_atr` | Макс. abs slope VWAP в ATR |
| `day_range_max_atr` | Макс. размер day range в ATR |
| `impulse_block_atr` | Блокировка при слишком сильном импульсе |
| `sweep_min_atr` | Мин. sweep в ATR |
| `sweep_max_atr` | Макс. sweep в ATR |
| `wick_min_share` | Мин. доля хвоста sweep-свечи |
| `sweep_volume_mult` | Мин. объем sweep-свечи |
| `return_close_distance_atr` | Допуск close при возврате в диапазон |
| `stop_buffer_atr` | Буфер стопа в ATR |
| `tp1_r` | TP1 в R |
| `tp2_r` | TP2 в R |
| `entry_mode` | Режим входа (`NEXT_BAR_OPEN` и т.д.) |

## Конфигурация `config/news_blackout.yaml`

Файл содержит список интервалов, внутри которых новые сигналы блокируются:

```yaml
- start: "2026-04-10 15:25"
  end: "2026-04-10 15:40"
  description: "CPI release"
```

Форматы времени: `YYYY-MM-DD HH:MM` или `YYYY-MM-DD HH:MM:SS`.  
Границы интервала включительные (`start <= now <= end`).

## Логи и данные

Логи пишутся в папку `logs/`:
- `application.log`
- `errors.log`
- `telegram.log`

SQLite (по умолчанию `signals.db`) содержит таблицы:
- `signals`
- `trades`
- `trade_events`
- `stats_snapshots`

## Запуск тестов

```powershell
python -m unittest discover -s tests -v
```

На текущей версии проходит 44 теста.

## Частые проблемы

- `MARKET_DATA_MODE=t_invest`, но пустой `INVEST_TOKEN`: приложение отправит critical alert и переключится в `demo`.
- Telegram включен, но пустые `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`: нотификатор отключится, торговая логика продолжит работу.
- Неверная структура YAML: приложение завершится с кодом ошибки конфигурации и отправит critical alert (если Telegram доступен).

## Важно

- Проект не отправляет реальные ордера на биржу.
- Все торговые параметры должны задаваться через конфиги, а не хардкодом.
