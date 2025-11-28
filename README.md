Resume Screening Agent Prototype

What this is:
- A minimal, local Resume Screening prototype that ranks text resumes against a job description using simple TF-based similarity and keyword hit ratio.

Files:
- `src/resume_screening.py` - main script to rank resumes
- `data/job_description.txt` - sample job description
- `data/resumes/*.txt` - sample resumes

How to run (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# No external deps required for this minimal prototype
python .\src\resume_screening.py
```

How it works (contract):
- Input: a job description text file and a folder of text resumes
- Output: a ranked list of resumes with similarity score, skill hit ratio, and suggested next steps

Limitations & next steps:
- Uses simple bag-of-words TF similarity (no semantic embeddings)
- Add embeddings (OpenAI/other) + vector DB for better matches
- Parse real PDF/DOCX resumes into text
- Integrate with UI (Streamlit) and store candidates in DB

Architecture (simple):

	[User uploads JD + Resumes] -> [Streamlit UI (src/app.py)] -> [Scoring Module (src/resume_screening.py)] -> [Ranking & Recommendations] -> [User]

Submission checklist
- Ensure all files are in the `resume_agent` folder
- Run unit tests: `python -m unittest resume_agent/tests/test_resume_screening.py -v`
- Package the folder as zip for upload

Resume Screening Agent Prototype

What this is:
- A minimal, local Resume Screening prototype that ranks text resumes against a job description using simple TF-based similarity and keyword hit ratio.

Files:
- `src/resume_screening.py` - main script to rank resumes
- `src/app.py` - Streamlit demo UI (accepts txt/pdf/docx)
- `src/parser.py` - PDF and DOCX text extraction helpers
- `data/job_description.txt` - sample job description
- `data/resumes/*.txt` - sample resumes
- `tests/*` - unit tests that validate core functions

How to run (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\resume_agent\requirements.txt
python -m unittest discover -v .\resume_agent\tests
```

Run the Streamlit demo:

```powershell
streamlit run .\resume_agent\src\app.py
```

How it works (contract):
- Input: a job description (txt/pdf/docx) and one or more resumes (txt/pdf/docx)
- Output: a ranked list of resumes with similarity score, skill hit ratio, matched keywords, and a brief recommendation

Limitations & next steps:
- Uses simple bag-of-words TF similarity (no semantic embeddings)
- Add embeddings (OpenAI/other) + vector DB for better matches
- OCR scanned PDFs (Tesseract) for image-only resumes
- Improve DOCX parsing (tables, sections), and add persistence (DB)

Architecture (simple):

    [User uploads JD + Resumes] -> [Streamlit UI (src/app.py)] -> [Parser (src/parser.py)] -> [Scoring Module (src/resume_screening.py)] -> [Ranking & Recommendations] -> [User]

Submission checklist
- Ensure all files are in the `resume_agent` folder
- Run unit tests: `python -m unittest discover -v .\resume_agent\tests`
- Package the folder as zip for upload: `Compress-Archive -Path .\resume_agent\* -DestinationPath resume_agent_submission.zip -Force`

Git push (optional)
- To create a local repo, commit, and push to GitHub (replace `<REMOTE_URL>`):

```powershell
cd .\resume_agent
git init
git add .
git commit -m "ResumeScreening agent prototype"
git remote add origin <REMOTE_URL>
git branch -M main
git push -u origin main
```

Contact / Notes
- This is a lightweight prototype for the ResumeScreening agent required in the challenge. For production, replace bag-of-words with embeddings and add resume parsing and OCR as needed.
