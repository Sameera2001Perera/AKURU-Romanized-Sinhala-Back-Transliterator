import unittest
from transliterator.transliteration import Transliterator


class TestTransliterator(unittest.TestCase):
    def setUp(self):
        self.transliterator = Transliterator("data/dictionary.txt", None, None)

    def test_simple_transliteration(self):
        result = self.transliterator.generate_sinhala("hello")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
