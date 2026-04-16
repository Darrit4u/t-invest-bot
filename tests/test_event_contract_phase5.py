from __future__ import annotations

import unittest

from core.portfolio_events import (
    DomainEventType,
    normalize_trade_event,
)
from core.trade_simulator import TradeEvent
from tests.helpers import dt_at


class EventContractPhase5Tests(unittest.TestCase):
    def test_activated_trade_event_is_normalized_to_position_opened(self) -> None:
        event = TradeEvent(
            trade_id="t1",
            signal_id="s1",
            instrument="ES",
            strategy="trend_pullback_vwap_ema",
            event_type="activated",
            status="activated",
            event_time=dt_at(1),
            price=101.0,
            size=1.0,
            payload={},
        )

        normalized = normalize_trade_event(event)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].kind, DomainEventType.POSITION_OPENED)
        self.assertEqual(normalized[0].event_type, "position_opened")

    def test_close_trade_event_is_normalized_to_position_and_trade_closed(self) -> None:
        event = TradeEvent(
            trade_id="t2",
            signal_id="s2",
            instrument="BRENT",
            strategy="compression_breakout",
            event_type="sl_hit",
            status="sl_hit",
            event_time=dt_at(2),
            price=80.2,
            size=0.0,
            payload={"reason": "stop_hit"},
        )

        normalized = normalize_trade_event(event)

        self.assertEqual(len(normalized), 2)
        kinds = {item.kind for item in normalized}
        self.assertEqual(
            kinds,
            {DomainEventType.POSITION_CLOSED, DomainEventType.TRADE_CLOSED},
        )


if __name__ == "__main__":
    unittest.main()
