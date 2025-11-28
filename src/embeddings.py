import os
import hashlib
import math
from typing import List

EMBED_DIM = 512


def _fallback_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    # Deterministic fallback: use SHA256 blocks to create floats in [-1,1]
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vals = []
    i = 0
    while len(vals) < dim:
        # extend hash with iteration
        hashi = hashlib.sha256(h + i.to_bytes(4, 'big')).digest()
        for b in hashi:
            if len(vals) >= dim:
                break
            # map byte to float in [-1,1]
            vals.append((b / 255.0) * 2 - 1)
        i += 1
    # normalize
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def get_embedding(text: str) -> List[float]:
    """Return embedding for text. Uses OpenAI if OPENAI_API_KEY set, otherwise a deterministic fallback."""
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        try:
            import openai
            openai.api_key = key
            resp = openai.Embedding.create(model='text-embedding-3-small', input=text)
            emb = resp['data'][0]['embedding']
            return emb
        except Exception:
            # fallback to deterministic embedding on errors
            return _fallback_embedding(text, dim=EMBED_DIM)
    return _fallback_embedding(text, dim=EMBED_DIM)
