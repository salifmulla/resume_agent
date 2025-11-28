import streamlit as st
from src.resume_screening import score_resume
from src.parser import extract_text_from_file_upload


st.set_page_config(page_title='Resume Screening Agent', layout='wide')
st.title('Resume Screening Agent — Demo')

st.markdown('Upload a job description (txt/pdf/docx) and either paste resumes or upload them. The agent ranks resumes by relevance.')

# Job description upload or paste
jd = st.file_uploader('Job description (txt/pdf/docx)', type=['txt', 'pdf', 'docx'])
# If user uploads a file, extract text and populate session state so the text_area shows it and is editable
if jd:
    extracted = extract_text_from_file_upload(jd)
    if extracted:
        st.session_state['job_text'] = extracted

# Provide an editable text area for the job description (pre-populated from upload if present)
if 'job_text' not in st.session_state:
    st.session_state['job_text'] = ''
job_text = st.text_area('Job description (paste or edit extracted text)', value=st.session_state.get('job_text', ''), key='job_text', height=200)

st.markdown('---')

# Resumes: allow a small number of paste areas and optional file upload per resume
num = st.number_input('How many resumes to evaluate', min_value=1, max_value=10, value=3)
resumes = []
for i in range(num):
    pasted = st.text_area(f'Resume {i+1} (paste text)', key=f'res_{i}', height=150)
    uploaded = st.file_uploader(f'Upload resume {i+1} (txt/pdf/docx)', type=['txt', 'pdf', 'docx'], key=f'upload_{i}')
    text = ''
    name = f'user_resume_{i+1}.txt'
    if uploaded:
        text = extract_text_from_file_upload(uploaded)
        name = getattr(uploaded, 'name', name)
    elif pasted and pasted.strip():
        text = pasted
    resumes.append((name, text))


use_embeddings = st.checkbox('Use semantic embeddings (OpenAI if API key present, otherwise deterministic fallback)', value=False)

if st.button('Run Screening'):
    if not job_text or job_text.strip() == '' or all(not r[1].strip() for r in resumes):
        st.error('Please provide a job description and at least one resume')
    else:
        results = []
        for name, text in resumes:
            if text and text.strip():
                r = score_resume(job_text, text, use_embeddings=use_embeddings)
                results.append({'resume': name, **r})
        if not results:
            st.warning('No valid resumes provided')
        else:
            results.sort(key=lambda x: (x['score'], x['skill_hit_ratio']), reverse=True)
            st.success('Ranking complete')
            for i, r in enumerate(results, 1):
                st.subheader(f"{i}. {r['resume']}")
                st.write(f"Score: {r['score']} — Skill hit ratio: {r['skill_hit_ratio']}")
                st.write('Matched keywords:', ', '.join(r['common_keywords'][:20]) if r['common_keywords'] else 'None')
                if r['skill_hit_ratio'] >= 0.6 and r['score'] > 0.05:
                    st.info('Recommendation: Strong match — invite for interview')
                elif r['skill_hit_ratio'] > 0:
                    st.warning('Recommendation: Partial match — technical screening suggested')
                else:
                    st.write('Recommendation: Low match')
