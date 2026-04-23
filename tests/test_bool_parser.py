from __future__ import annotations

import unittest

from core.bool_parser import to_bool


class BoolParserTests(unittest.TestCase):
    def test_parses_typical_string_values(self) -> None:
        cases = (
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
            ("  TRUE  ", True),
            ("  Off ", False),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(to_bool(raw, default=False), expected)

    def test_keeps_bool_and_int_inputs(self) -> None:
        cases = (
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (2, True),
            (-1, True),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(to_bool(raw, default=False), expected)

    def test_uses_default_for_unrecognized_values(self) -> None:
        self.assertTrue(to_bool("not-a-bool", default=True))
        self.assertFalse(to_bool("not-a-bool", default=False))
        self.assertTrue(to_bool(None, default=True))
        self.assertFalse(to_bool(None, default=False))


if __name__ == "__main__":
    unittest.main()
