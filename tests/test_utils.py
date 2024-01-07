import unittest
from chlamy_impi.database_creation.utils import parse_name


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

    def test_parse_name_invalid_input(self):
        with self.assertRaises(AssertionError):
            parse_name("invalid_filename.npy")

if __name__ == '__main__':
    unittest.main()
