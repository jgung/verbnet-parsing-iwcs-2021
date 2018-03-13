import unittest

from tfnlp.common.utils import read_jsons


class TestUtils(unittest.TestCase):
    def test_read_json(self):
        test_json = "{\"foo\": 1, \"bar\": {\"foo\": 2, \"bar\": [{\"foobar\": 3}]}}"
        result = read_jsons(test_json)
        self.assertEqual(1, result.foo)
        self.assertEqual(2, result.bar.foo)
        self.assertEqual(3, result.bar.bar[0].foobar)
