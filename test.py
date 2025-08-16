import unittest

from solar import add, State


class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)


class TestState(unittest.TestCase):
    def test_state_values(self):
        """Test that state enum values are correct abbreviations"""
        self.assertEqual(State.NEW_YORK.value, "ny")
        self.assertEqual(State.TEXAS.value, "tx")
        self.assertEqual(State.NEW_MEXICO_EAST.value, "nm-east")

    def test_full_name(self):
        """Test full_name property returns properly formatted names"""
        self.assertEqual(State.CALIFORNIA.full_name(), "California")
        self.assertEqual(State.NEW_YORK.full_name(), "New York")
        self.assertEqual(State.NEW_MEXICO_EAST.full_name(), "New Mexico East")

    def test_from_str_by_abbreviation(self):
        """Test from_str method with state abbreviations"""
        self.assertEqual(State("ca"), State.CALIFORNIA)
        self.assertEqual(State("ny"), State.NEW_YORK)
        self.assertEqual(State("tx-east"), State.TEXAS_EAST)
        self.assertEqual(State.from_str("CA"), State.CALIFORNIA)
        self.assertEqual(State.from_str("ny"), State.NEW_YORK)
        self.assertEqual(State.from_str("nm-east"), State.NEW_MEXICO_EAST)

    def test_from_str_by_full_name(self):
        """Test from_str method with full state names"""
        self.assertEqual(State.from_str("New York"), State.NEW_YORK)
        self.assertEqual(State.from_str("new york"), State.NEW_YORK)

    def test_from_str_invalid(self):
        """Test from_str raises ValueError for invalid inputs"""
        with self.assertRaises(ValueError):
            State.from_str("invalid")


if __name__ == "__main__":
    unittest.main()
