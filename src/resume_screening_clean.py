import re
import math
from pathlib import Path
from collections import Counter
from typing import List

STOPWORDS = {
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','which'
}


def preprocess(text: str) -> List[str]:
    text = (text or '').lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]


def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    c = Counter(tokens)
    return [float(c.get(w, 0)) for w in vocab]


def l2_norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
    if use_embeddings:
        try:
            from src.embeddings import get_embedding
            ja = get_embedding(job_text)
            ra = get_embedding(resume_text)
            num = sum(x * y for x, y in zip(ja, ra))
            den_a = math.sqrt(sum(x * x for x in ja)) or 1.0
            den_b = math.sqrt(sum(x * x for x in ra)) or 1.0
            sim = num / (den_a * den_b)
        except Exception:
            sim = 0.0
    else:
        jt = preprocess(job_text)
        rt = preprocess(resume_text)
        vocab = sorted(set(jt) | set(rt))
        vj = l2_norm(tf_vector(jt, vocab))
        vr = l2_norm(tf_vector(rt, vocab))
        sim = sum(x * y for x, y in zip(vj, vr))

    jd_words = set(preprocess(job_text))
    common = sorted(jd_words & set(preprocess(resume_text)))
    skill_hit = float(len(common)) / (len(jd_words) or 1)
    return {'score': float(round(sim, 6)), 'skill_hit_ratio': float(round(skill_hit, 3)), 'common_keywords': common[:20]}


def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = [(p.name, p.read_text(encoding='utf-8')) for p in sorted(Path(resumes_dir).glob('*.txt'))]
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text, use_embeddings=use_embeddings)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results
