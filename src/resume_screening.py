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
    for i,r in enumerate(results,1):
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
