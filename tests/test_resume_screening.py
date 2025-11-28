import unittest
from pathlib import Path
import importlib.util
import sys

# Load module by file path to ensure tests run regardless of PYTHONPATH
spec = importlib.util.spec_from_file_location('src.resume_screening_clean', str(Path(__file__).resolve().parents[1] / 'src' / 'resume_screening_clean.py'))
resume_mod = importlib.util.module_from_spec(spec)
sys.modules['src.resume_screening_clean'] = resume_mod
spec.loader.exec_module(resume_mod)
score_resume = resume_mod.score_resume
rank_resumes = resume_mod.rank_resumes

class TestResumeScreening(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]
        self.job = base / 'data' / 'job_description.txt'
        self.resumes_dir = base / 'data' / 'resumes'
        self.job_text = self.job.read_text(encoding='utf-8')
    def test_score_returns_keys(self):
        r = score_resume(self.job_text, (self.resumes_dir / 'alice_engineer.txt').read_text())
        self.assertIn('score', r)
        self.assertIn('skill_hit_ratio', r)
        self.assertIn('common_keywords', r)
    def test_rank_order(self):
        results = rank_resumes(self.job, self.resumes_dir)
        self.assertTrue(results[0]['score'] >= results[1]['score'])

    def test_embeddings_fallback(self):
        # ensure embeddings path runs (uses deterministic fallback when no API key)
        r = score_resume(self.job_text, (self.resumes_dir / 'alice_engineer.txt').read_text(), use_embeddings=True)
        self.assertIn('score', r)
        self.assertIsInstance(r['score'], float)

if __name__ == '__main__':
    unittest.main()
