import unittest
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('src.embeddings', str(Path(__file__).resolve().parents[1] / 'src' / 'embeddings.py'))
emb_mod = importlib.util.module_from_spec(spec)
sys.modules['src.embeddings'] = emb_mod
spec.loader.exec_module(emb_mod)
get_embedding = emb_mod.get_embedding


class TestEmbeddings(unittest.TestCase):
    def test_dimensions_and_repeatability(self):
        a = get_embedding('machine learning engineer')
        b = get_embedding('machine learning engineer')
        self.assertEqual(len(a), len(b))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 100)


if __name__ == '__main__':
    unittest.main()
