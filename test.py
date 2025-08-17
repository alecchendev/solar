import unittest

from solar import (
    Command,
    DataType,
    DatasetColumn,
    PvType,
    State,
    metadata_from_filename,
    solar_filename,
)


class TestState(unittest.TestCase):
    def test_full_name(self):
        self.assertEqual(State.CALIFORNIA.full_name(), "California")
        self.assertEqual(State.NEW_YORK.full_name(), "New York")
        self.assertEqual(State.NEW_MEXICO_EAST.full_name(), "New Mexico East")

    def test_from_str_by_abbreviation(self):
        self.assertEqual(State("ca"), State.CALIFORNIA)
        self.assertEqual(State("ny"), State.NEW_YORK)
        self.assertEqual(State("tx-east"), State.TEXAS_EAST)
        self.assertEqual(State.from_str("CA"), State.CALIFORNIA)
        self.assertEqual(State.from_str("ny"), State.NEW_YORK)
        self.assertEqual(State.from_str("nm-east"), State.NEW_MEXICO_EAST)

    def test_from_str_by_full_name(self):
        self.assertEqual(State.from_str("New York"), State.NEW_YORK)
        self.assertEqual(State.from_str("new york"), State.NEW_YORK)

    def test_from_str_invalid(self):
        with self.assertRaises(ValueError):
            State.from_str("invalid")

    def test_enums_display_strs(self):
        self.assertEqual(str(State.ARKANSAS), "ar")
        self.assertEqual(str(DataType.ACTUAL), "Actual")
        self.assertEqual(str(PvType.UPV), "UPV")
        self.assertEqual(str(DatasetColumn.STATE), "state")
        self.assertEqual(str(Command.DOWNLOAD), "download")

    def test_enums_are_strs(self):
        self.assertEqual(DatasetColumn.STATE, "state")
        self.assertEqual(DataType.ACTUAL, "Actual")
        self.assertEqual(PvType.UPV, "UPV")
        self.assertEqual(Command.DOWNLOAD, "download")

    def test_metadata_parse(self):
        example_filename = "Actual_33.45_-112.15_2006_DPV_103MW_5_Min.csv"
        metadata = metadata_from_filename(State.IOWA, example_filename)
        self.assertEqual(metadata[DatasetColumn.DATA_TYPE], DataType.ACTUAL)
        self.assertEqual(metadata[DatasetColumn.LATITUDE], 33.45)
        self.assertEqual(metadata[DatasetColumn.LONGITUDE], -112.15)
        self.assertEqual(metadata[DatasetColumn.WEATHER_YEAR], 2006)
        self.assertEqual(metadata[DatasetColumn.CAPACITY_MW], 103)
        self.assertEqual(metadata[DatasetColumn.TIME_INTERVAL_MIN], 5)
        self.assertEqual(solar_filename(**metadata), example_filename)

        example_float_cap_filename = "DA_43.05_-114.85_2006_UPV_0.2MW_60_Min.csv"
        metadata_float_cap = metadata_from_filename(
            State.IOWA, example_float_cap_filename
        )
        self.assertEqual(metadata_float_cap[DatasetColumn.CAPACITY_MW], 0.2)


if __name__ == "__main__":
    unittest.main()
