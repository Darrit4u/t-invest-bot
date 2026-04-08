from typing import Dict

class StatsTracker:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, int]] = {}

    def _ensure(self, alias: str) -> None:
        if alias not in self.data:
            self.data[alias] = {
                "entries": 0,
                "tp1": 0,
                "tp2": 0,
                "sl": 0,
                "breakeven": 0,
            }

    def record_entry(self, alias: str) -> None:
        self._ensure(alias)
        self.data[alias]["entries"] += 1

    def record_tp1(self, alias: str) -> None:
        self._ensure(alias)
        self.data[alias]["tp1"] += 1

    def record_tp2(self, alias: str) -> None:
        self._ensure(alias)
        self.data[alias]["tp2"] += 1

    def record_sl(self, alias: str) -> None:
        self._ensure(alias)
        self.data[alias]["sl"] += 1

    def record_breakeven(self, alias: str) -> None:
        self._ensure(alias)
        self.data[alias]["breakeven"] += 1

    def build_report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 72)
        lines.append("ИТОГОВАЯ СТАТИСТИКА ПО АЛГОРИТМУ")
        lines.append("=" * 72)

        total_entries = 0
        total_tp1 = 0
        total_tp2 = 0
        total_sl = 0
        total_be = 0

        for alias, s in self.data.items():
            entries = s["entries"]
            tp1 = s["tp1"]
            tp2 = s["tp2"]
            sl = s["sl"]
            breakeven = s["breakeven"]

            total_entries += entries
            total_tp1 += tp1
            total_tp2 += tp2
            total_sl += sl
            total_be += breakeven

            closed = tp2 + sl + breakeven
            tp1_rate = (tp1 / entries * 100) if entries else 0.0
            tp2_rate = (tp2 / entries * 100) if entries else 0.0
            sl_rate = (sl / entries * 100) if entries else 0.0

            # Успешность считаем в двух вариантах:
            # 1) Строгая: только полностью закрытые по TP2
            strict_success = (tp2 / closed * 100) if closed else 0.0
            # 2) Мягкая: TP2 + безубыток против SL
            protected_success = ((tp2 + breakeven) / closed * 100) if closed else 0.0

            lines.append(f"\nИнструмент: {alias}")
            lines.append(f"  Входов: {entries}")
            lines.append(f"  TP1: {tp1}")
            lines.append(f"  TP2 (полное закрытие): {tp2}")
            lines.append(f"  SL: {sl}")
            lines.append(f"  Безубыток: {breakeven}")
            lines.append(f"  TP1 rate: {tp1_rate:.2f}%")
            lines.append(f"  TP2 rate: {tp2_rate:.2f}%")
            lines.append(f"  SL rate: {sl_rate:.2f}%")
            lines.append(f"  Успешность (строгая, TP2/все закрытые): {strict_success:.2f}%")
            lines.append(f"  Успешность (TP2+BE против SL): {protected_success:.2f}%")
            lines.append(f"  Соотношение TP2/SL: {tp2}:{sl}")

        total_closed = total_tp2 + total_sl + total_be
        total_tp1_rate = (total_tp1 / total_entries * 100) if total_entries else 0.0
        total_tp2_rate = (total_tp2 / total_entries * 100) if total_entries else 0.0
        total_sl_rate = (total_sl / total_entries * 100) if total_entries else 0.0
        total_strict_success = (total_tp2 / total_closed * 100) if total_closed else 0.0
        total_protected_success = ((total_tp2 + total_be) / total_closed * 100) if total_closed else 0.0

        lines.append("\n" + "-" * 72)
        lines.append("ОБЩИЙ ИТОГ")
        lines.append("-" * 72)
        lines.append(f"Всего входов: {total_entries}")
        lines.append(f"Всего TP1: {total_tp1}")
        lines.append(f"Всего TP2: {total_tp2}")
        lines.append(f"Всего SL: {total_sl}")
        lines.append(f"Всего безубытков: {total_be}")
        lines.append(f"Общий TP1 rate: {total_tp1_rate:.2f}%")
        lines.append(f"Общий TP2 rate: {total_tp2_rate:.2f}%")
        lines.append(f"Общий SL rate: {total_sl_rate:.2f}%")
        lines.append(f"Общая успешность (строгая): {total_strict_success:.2f}%")
        lines.append(f"Общая успешность (TP2+BE против SL): {total_protected_success:.2f}%")
        lines.append(f"Общее соотношение TP2/SL: {total_tp2}:{total_sl}")
        lines.append("=" * 72)

        return "\n".join(lines)