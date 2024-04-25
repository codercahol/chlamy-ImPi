import unittest
from datetime import datetime

from chlamy_impi.database_creation.utils import parse_name, spreadsheet_plate_to_numeric


class TestSpreadsheetPlateToNumeric(unittest.TestCase):

        def test_spreadsheet_plate_to_numeric_valid_input_1(self):
            result = spreadsheet_plate_to_numeric("Plate 01")
            self.assertEqual(result, 1)

        def test_spreadsheet_plate_to_numeric_valid_input_2(self):
            result = spreadsheet_plate_to_numeric("Plate 12")
            self.assertEqual(result, 12)

        def test_spreadsheet_plate_to_numeric_valid_input_3(self):
            result = spreadsheet_plate_to_numeric("Plate 99")
            self.assertEqual(result, 99)

        def test_spreadsheet_plate_to_numeric_invalid_input(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 0")

        def test_spreadsheet_plate_to_numeric_invalid_input_2(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 100")

        def test_spreadsheet_plate_to_numeric_invalid_input_3(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 1a")

        def test_spreadsheet_plate_to_numeric_invalid_input_4(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 01a")

        def test_spreadsheet_plate_to_numeric_invalid_input_5(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 1")

        def test_spreadsheet_plate_to_numeric_invalid_input_6(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 001")

        def test_spreadsheet_plate_to_numeric_invalid_input_7(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_to_numeric("Plate 1.0")


class TestParseName(unittest.TestCase):

    def test_parse_name_valid_input_1(self):
        result = parse_name("20200303_7-M4_2h-2h.npy")
        self.assertEqual(result, (7, 'M4', '2h-2h'))

    def test_parse_name_valid_input_2(self):
        result = parse_name("20231119_07-M3_20h_ML.npy")
        self.assertEqual(result, (7, 'M3', '20h_ML'))

    def test_parse_name_valid_input_3(self):
        result = parse_name("20231213_9-M5_2h-2h.npy")
        self.assertEqual(result, (9, 'M5', '2h-2h'))

    def test_parse_name_valid_input_4(self):
        result = parse_name("20231213_99-M5_2h-2h.npy")
        self.assertEqual(result, (99, 'M5', '2h-2h'))

    def test_parse_name_valid_input_5(self):
        result = parse_name("20231213_99-M5_2h-2h.npy", return_date=True)
        self.assertEqual(result, (99, 'M5', '2h-2h', datetime(year=2023, month=12, day=13)))

    def test_parse_name_invalid_input(self):
        with self.assertRaises(AssertionError):
            parse_name("invalid_filename.npy")

if __name__ == '__main__':
    unittest.main()
