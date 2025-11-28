from io import BytesIO
from pathlib import Path


def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError('PyPDF2 is required to parse PDFs') from e
    reader = PdfReader(BytesIO(b))
    texts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        import docx
    except Exception as e:
        raise RuntimeError('python-docx is required to parse DOCX') from e
    from io import BytesIO
    doc = docx.Document(BytesIO(b))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n\n".join(paras)


def extract_text_from_file_upload(upload) -> str:
    # upload: a BytesIO-like object from Streamlit
    if not upload:
        return ''
    name = getattr(upload, 'name', '')
    data = upload.read()
    if name.lower().endswith('.pdf'):
        return extract_text_from_pdf_bytes(data)
    if name.lower().endswith('.docx'):
        return extract_text_from_docx_bytes(data)
    # fallback: attempt to decode as utf-8 text
    try:
        return data.decode('utf-8')
    except Exception:
        return ''
