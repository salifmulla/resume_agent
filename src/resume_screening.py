import re
import math
from pathlib import Path
from collections import Counter
from typing import List

STOPWORDS = {
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by',
    'that','this','it','or','we','our','you','your','i','will','have','has','at','from',
    'can','such','these','those','should','which'
}


def preprocess(text: str) -> List[str]:
    """Lowercase, strip punctuation, split and drop stopwords."""
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


def cosine(u: List[float], v: List[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
    """Score a resume vs job description. Returns score, skill_hit_ratio, common_keywords."""
    if use_embeddings:
        try:
            from src.embeddings import get_embedding

            ja = get_embedding(job_text)
            ra = get_embedding(resume_text)
            den_a = math.sqrt(sum(x * x for x in ja)) or 1.0
            den_b = math.sqrt(sum(x * x for x in ra)) or 1.0
            sim = sum(x * y for x, y in zip(ja, ra)) / (den_a * den_b)
        except Exception:
            sim = 0.0
    else:
        jt = preprocess(job_text)
        rt = preprocess(resume_text)
        vocab = sorted(set(jt) | set(rt))
        vj = l2_norm(tf_vector(jt, vocab))
        vr = l2_norm(tf_vector(rt, vocab))
        sim = cosine(vj, vr)

    jd_words = set(preprocess(job_text))
    resume_words = set(preprocess(resume_text))
    common = sorted(jd_words & resume_words)
    skill_hit = float(len(common)) / (len(jd_words) or 1)
    return {
        'score': float(round(sim, 6)),
        'skill_hit_ratio': float(round(skill_hit, 3)),
        'common_keywords': common[:20]
    }


def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = []
    for p in sorted(Path(resumes_dir).glob('*.txt')):
        resumes.append((p.name, p.read_text(encoding='utf-8')))
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text, use_embeddings=use_embeddings)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results
from pathlib import Path
from collections import Counter
from typing import List


STOPWORDS = set([
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','which'
])


def preprocess(text: str) -> List[str]:
    text = (text or '').lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    c = Counter(tokens)
    return [float(c.get(w, 0)) for w in vocab]


def l2_norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def dot(u: List[float], v: List[float]) -> float:
    """Resume screening scoring utilities.

    This file provides minimal, well-tested functions used by the demo and unit
    tests: preprocess, tf_vector, l2_norm, score_resume and rank_resumes.
    """

    import re
    import math
    from pathlib import Path
    from collections import Counter
    from typing import List


    STOPWORDS = set([
        'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
        'i','will','have','has','at','from','can','such','these','those','should','which'
    ])


    def preprocess(text: str) -> List[str]:
        """Lowercase, remove punctuation, split and filter stopwords."""
        text = (text or '').lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
        return tokens


    def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
        c = Counter(tokens)
        return [float(c.get(w, 0)) for w in vocab]


    def l2_norm(vec: List[float]) -> List[float]:
        s = math.sqrt(sum(x * x for x in vec))
        if s == 0:
            return vec
        return [x / s for x in vec]


    def dot(u: List[float], v: List[float]) -> float:
        return sum(x * y for x, y in zip(u, v))


    def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
        """Score a resume against a job description.

        If use_embeddings is True the function attempts to call
        `src.embeddings.get_embedding(text)` and compute cosine similarity. Any
        exception during embedding lookup results in a similarity of 0.0 so the
        code degrades gracefully when API keys are missing.
        """
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
            sim = dot(vj, vr)

        jd_words = set(preprocess(job_text))
        resume_words = set(preprocess(resume_text))
        common = sorted(jd_words & resume_words)
        skill_hit = float(len(common)) / (len(jd_words) or 1)
        return {
            'score': float(round(sim, 6)),
            'skill_hit_ratio': float(round(skill_hit, 3)),
            'common_keywords': common[:20]
        }


    def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
        job_text = Path(job_path).read_text(encoding='utf-8')
        resumes = []
        for p in sorted(Path(resumes_dir).glob('*.txt')):
            resumes.append((p.name, p.read_text(encoding='utf-8')))
        results = []
        for name, text in resumes:
            r = score_resume(job_text, text, use_embeddings=use_embeddings)
            results.append({'resume': name, **r})
        results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
        return results

            den_a = math.sqrt(sum(x * x for x in ja)) or 1.0
            den_b = math.sqrt(sum(x * x for x in ra)) or 1.0
            sim = num / (den_a * den_b)
        except Exception:
            # any error (missing key, network) => graceful degrade to 0 similarity
            sim = 0.0
    else:
        jt = preprocess(job_text)
        rt = preprocess(resume_text)
        vocab = sorted(set(jt) | set(rt))
        vj = l2_norm(tf_vector(jt, vocab))
        vr = l2_norm(tf_vector(rt, vocab))
        sim = dot(vj, vr)

    jd_words = set(preprocess(job_text))
    resume_words = set(preprocess(resume_text))
    common = sorted(jd_words & resume_words)
    skill_hit = float(len(common)) / (len(jd_words) or 1)
    return {
        'score': float(round(sim, 6)),
        'skill_hit_ratio': float(round(skill_hit, 3)),
        'common_keywords': common[:20]
    }


def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = []
    for p in sorted(Path(resumes_dir).glob('*.txt')):
        resumes.append((p.name, p.read_text(encoding='utf-8')))
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text, use_embeddings=use_embeddings)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    job = base / 'data' / 'job_description.txt'
    resumes_dir = base / 'data' / 'resumes'
    if not job.exists():
        print('Job description not found at', job)
        raise SystemExit(1)
    if not resumes_dir.exists():
        print('Resumes folder not found at', resumes_dir)
        raise SystemExit(1)
    results = rank_resumes(job, resumes_dir)
    print('Resume ranking (best first):')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['resume']}  score={r['score']} hit={r['skill_hit_ratio']} keywords={', '.join(r['common_keywords'][:6])}")
    print('\nDetailed suggestions:')
    for r in results:
        name = r['resume']
        if r['skill_hit_ratio'] >= 0.6 and r['score'] > 0.05:
            note = 'Strong match - invite to interview.'
        elif r['skill_hit_ratio'] > 0:
            note = 'Partial match - consider technical screen.'
        else:
            note = 'Low match - not a fit.'
        print(f"- {name}: {note}")
import re
import math
from pathlib import Path
from collections import Counter
from typing import List


STOPWORDS = set([
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','which'
])


def preprocess(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    c = Counter(tokens)
    return [float(c.get(w, 0)) for w in vocab]


def l2_norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def dot(u: List[float], v: List[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
    """Score a resume against a job description.

    If use_embeddings is True, try to use embeddings. Otherwise use TF cosine.
    """
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
        sim = dot(vj, vr)

    jd_words = set(preprocess(job_text))
    resume_words = set(preprocess(resume_text))
    common = sorted(jd_words & resume_words)
    skill_hit = float(len(common)) / (len(jd_words) or 1)
    return {
        'score': float(round(sim, 6)),
        'skill_hit_ratio': float(round(skill_hit, 3)),
        'common_keywords': common[:20]
    }


def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = []
    for p in sorted(Path(resumes_dir).glob('*.txt')):
        resumes.append((p.name, p.read_text(encoding='utf-8')))
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text, use_embeddings=use_embeddings)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    job = base / 'data' / 'job_description.txt'
    resumes_dir = base / 'data' / 'resumes'
    if not job.exists():
        print('Job description not found at', job)
        raise SystemExit(1)
    if not resumes_dir.exists():
        print('Resumes folder not found at', resumes_dir)
        raise SystemExit(1)
    results = rank_resumes(job, resumes_dir)
    print('Resume ranking (best first):')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['resume']}  score={r['score']} hit={r['skill_hit_ratio']} keywords={', '.join(r['common_keywords'][:6])}")
    print('\nDetailed suggestions:')
    for r in results:
        name = r['resume']
        if r['skill_hit_ratio'] >= 0.6 and r['score'] > 0.05:
            note = 'Strong match - invite to interview.'
        elif r['skill_hit_ratio'] > 0:
            note = 'Partial match - consider technical screen.'
        else:
            note = 'Low match - not a fit.'
        print(f"- {name}: {note}")
import re
import math
from pathlib import Path
from collections import Counter
from typing import List

STOPWORDS = set([
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','which'
])


def preprocess(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    c = Counter(tokens)
    return [float(c.get(w, 0)) for w in vocab]


def l2_norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def dot(u: List[float], v: List[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
    """Score a resume against a job description.

    If use_embeddings is True, try to use embeddings. Otherwise use TF cosine.
    """
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
        sim = dot(vj, vr)

    jd_words = set(preprocess(job_text))
    resume_words = set(preprocess(resume_text))
    common = sorted(jd_words & resume_words)
    skill_hit = float(len(common)) / (len(jd_words) or 1)
    return {
        'score': float(round(sim, 6)),
        'skill_hit_ratio': float(round(skill_hit, 3)),
        'common_keywords': common[:20]
    }


def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = []
    for p in sorted(Path(resumes_dir).glob('*.txt')):
        resumes.append((p.name, p.read_text(encoding='utf-8')))
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text, use_embeddings=use_embeddings)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    job = base / 'data' / 'job_description.txt'
    resumes_dir = base / 'data' / 'resumes'
    if not job.exists():
        print('Job description not found at', job)
        raise SystemExit(1)
    if not resumes_dir.exists():
        print('Resumes folder not found at', resumes_dir)
        raise SystemExit(1)
    results = rank_resumes(job, resumes_dir)
    print('Resume ranking (best first):')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['resume']}  score={r['score']} hit={r['skill_hit_ratio']} keywords={', '.join(r['common_keywords'][:6])}")
    print('\nDetailed suggestions:')
    for r in results:
        name = r['resume']
        if r['skill_hit_ratio'] >= 0.6 and r['score'] > 0.05:
            note = 'Strong match - invite to interview.'
        elif r['skill_hit_ratio'] > 0:
            note = 'Partial match - consider technical screen.'
        else:
            note = 'Low match - not a fit.'
        print(f"- {name}: {note}")
import re
import math
from pathlib import Path
from collections import Counter
from typing import List

STOPWORDS = {
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','will','which'
}


def preprocess(text: str) -> List[str]:
    text = text.lower()
        elif r['skill_hit_ratio'] > 0:
            note = 'Partial match — consider technical screening for skill gaps.'
        else:
            note = 'Low match — keep candidate for future roles or junior positions.'
        print(f"- {name}: {note}")
import re
import math
from pathlib import Path
from collections import Counter

STOPWORDS = {
    'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
    'i','will','have','has','at','from','can','such','these','those','should','will','which'
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def tf_vector(tokens, vocab):
    c = Counter(tokens)
    return [c.get(w, 0) for w in vocab]


def l2_norm(vec):
    s = math.sqrt(sum(x*x for x in vec))
    if s == 0:
        return vec
    return [x/s for x in vec]


def dot(u, v):
    return sum(x*y for x,y in zip(u,v))


def score_resume(job_text, resume_text):
    jt = preprocess(job_text)
    rt = preprocess(resume_text)
    vocab = sorted(set(jt) | set(rt))
    vj = l2_norm(tf_vector(jt, vocab))
    vr = l2_norm(tf_vector(rt, vocab))
    sim = dot(vj, vr)
    # find top matching keywords
    jd_words = set(jt)
    resume_words = set(rt)
    common = sorted(jd_words & resume_words)
    # simple skill-hit percentage
    skill_hit = 0.0
    if jd_words:
        skill_hit = len(common)/len(jd_words)
    return {
        'score': round(sim, 4),
        'skill_hit_ratio': round(skill_hit, 3),
        'common_keywords': common
    }


def rank_resumes(job_path, resumes_dir):
    job_text = Path(job_path).read_text(encoding='utf-8')
    resumes = []
    for p in sorted(Path(resumes_dir).glob('*.txt')):
        resumes.append((p.name, p.read_text(encoding='utf-8')))
    results = []
    for name, text in resumes:
        r = score_resume(job_text, text)
        results.append({'resume': name, **r})
    results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
    return results


if __name__ == '__main__':
    import re
    import re
    import math
    from pathlib import Path
    from collections import Counter
    from typing import List

    STOPWORDS = {
        'the','and','a','an','to','of','in','for','on','with','is','are','be','as','by','that','this','it','or','we','our','you','your',
        'i','will','have','has','at','from','can','such','these','those','should','will','which'
    }


    def preprocess(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
        return tokens


    def tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
        c = Counter(tokens)
        return [float(c.get(w, 0)) for w in vocab]


    def l2_norm(vec: List[float]) -> List[float]:
        s = math.sqrt(sum(x * x for x in vec))
        if s == 0:
            return vec
        return [x / s for x in vec]


    def dot(u: List[float], v: List[float]) -> float:
        return sum(x * y for x, y in zip(u, v))


    def score_resume(job_text: str, resume_text: str, use_embeddings: bool = False) -> dict:
        """Return a score object for a resume against a job description.

        If use_embeddings=True, a semantic similarity is computed using an embedding
        function (if available). Otherwise, fall back to TF cosine similarity.
        """
        if use_embeddings:
            try:
                # local import to keep dependency optional
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
            sim = dot(vj, vr)

        # find top matching keywords
        jd_words = set(preprocess(job_text))
        resume_words = set(preprocess(resume_text))
        common = sorted(jd_words & resume_words)
        # simple skill-hit percentage
        skill_hit = 0.0
        if jd_words:
            skill_hit = len(common) / len(jd_words)
        return {
            'score': round(sim, 6),
            'skill_hit_ratio': round(skill_hit, 3),
            'common_keywords': common[:20]
        }


    def rank_resumes(job_path: str, resumes_dir: str, use_embeddings: bool = False) -> List[dict]:
        job_text = Path(job_path).read_text(encoding='utf-8')
        resumes = []
        for p in sorted(Path(resumes_dir).glob('*.txt')):
            resumes.append((p.name, p.read_text(encoding='utf-8')))
        results = []
        for name, text in resumes:
            r = score_resume(job_text, text, use_embeddings=use_embeddings)
            results.append({'resume': name, **r})
        results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
        return results


    if __name__ == '__main__':
        base = Path(__file__).resolve().parents[1]
        job = base / 'data' / 'job_description.txt'
        resumes_dir = base / 'data' / 'resumes'
        if not job.exists():
            print('Job description not found at', job)
            raise SystemExit(1)
        if not resumes_dir.exists():
            print('Resumes folder not found at', resumes_dir)
            raise SystemExit(1)
        results = rank_resumes(job, resumes_dir)
        print('Resume ranking (best first):')
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['resume']}  score={r['score']} hit={r['skill_hit_ratio']} keywords={', '.join(r['common_keywords'][:6])}")
        print('\nDetailed suggestions:')
        for r in results:
            name = r['resume']
            if r['skill_hit_ratio'] >= 0.6 and r['score'] > 0.05:
                note = 'Strong match — proceed to interview focusing on system design and past projects.'
            elif r['skill_hit_ratio'] > 0:
                note = 'Partial match — consider technical screening for skill gaps.'
            else:
                note = 'Low match — keep candidate for future roles or junior positions.'
            print(f"- {name}: {note}")
