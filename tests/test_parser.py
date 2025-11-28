import unittest
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('parser', str(Path(__file__).resolve().parents[1] / 'src' / 'parser.py'))
parser = importlib.util.module_from_spec(spec)
sys.modules['parser'] = parser
spec.loader.exec_module(parser)

class TestParser(unittest.TestCase):
    def test_functions_exist(self):
        self.assertTrue(hasattr(parser, 'extract_text_from_pdf_bytes'))
        self.assertTrue(hasattr(parser, 'extract_text_from_docx_bytes'))
        self.assertTrue(hasattr(parser, 'extract_text_from_file_upload'))

if __name__ == '__main__':
    unittest.main()
