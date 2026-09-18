import io
import json
import os
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def extract_text(uploaded_file) -> str:
    """Extract readable text from PDF, DOCX, or TXT resume files."""
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if file_name.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs).strip()

    if file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def analyze_resume(resume_text: str, job_description: str = "") -> dict:
    """Ask Gemini to evaluate the resume and return structured ATS feedback with retry logic."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets or your environment."
        )

    client = genai.Client(api_key=api_key)

    job_context = job_description.strip() or (
        "No job description was provided. Evaluate the resume using general ATS best practices."
    )

    prompt = f"""
You are an expert ATS resume reviewer and career coach.

Analyze the resume below. Give a practical ATS-oriented evaluation, but clearly state
that an ATS score is an estimate rather than a real score from a specific company's ATS.

Return ONLY valid JSON matching this exact structure:
{{
  "ats_score": 0,
  "summary": "short overall assessment",
  "section_scores": {{
    "format_and_parsability": 0,
    "keyword_optimization": 0,
    "experience_and_impact": 0,
    "skills_alignment": 0,
    "clarity_and_readability": 0
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": [
    {{
      "priority": "High",
      "issue": "specific issue",
      "recommendation": "specific fix"
    }}
  ],
  "missing_keywords": ["keyword 1", "keyword 2"],
  "rewrites": [
    {{
      "before": "weak or unclear resume wording",
      "after": "stronger ATS-friendly wording"
    }}
  ],
  "ats_checklist": {{
    "standard_headings": true,
    "simple_structure": true,
    "measurable_achievements": false,
    "relevant_keywords": true,
    "contact_information_present": true
  }}
}}

Rules:
- ats_score and every section score must be an integer from 0 to 100.
- Do not invent facts about the candidate.
- Base keyword recommendations on the provided job description when available.
- Prefer measurable, truthful improvements.
- Identify likely ATS parsing problems such as tables, columns, graphics, icons,
  unusual headings, headers/footers, or overly complex formatting only when there
  is evidence in the extracted text or the user explicitly describes them.
- Keep the response concise but useful.
- If the resume is too short or extraction looks incomplete, mention that in summary.
- Do not use markdown outside JSON.

JOB DESCRIPTION:
{job_context}

RESUME:
{resume_text[:30000]}
"""

    # Retry logic for handling 503 errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            st.info(f"🔄 Analyzing with Gemini (Attempt {attempt + 1}/{max_retries})...")
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )

            raw = (response.text or "").strip()
            if not raw:
                raise RuntimeError("Gemini returned an empty response.")

            try:
                result = json.loads(raw)
                st.success(f"✅ Analysis complete!")
                return result
            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemini returned invalid JSON. Please try again.") from exc

        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a 503/rate limit error
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "overloaded" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    st.warning(f"⏳ Server busy (503)... Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        "Gemini API is currently overloaded. Please try again in a few minutes."
                    )
            else:
                # Other error - don't retry
                raise


def show_results(result: dict) -> None:
    score = int(result.get("ats_score", 0))
    score = max(0, min(100, score))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Estimated ATS Score", f"{score}/100")
    with col2:
        st.progress(score / 100)
        st.caption("This is an AI estimate, not a score from a specific employer's ATS.")

    st.subheader("Overall assessment")
    st.write(result.get("summary", "No summary returned."))

    st.subheader("Section scores")
    section_scores = result.get("section_scores", {})
    score_cols = st.columns(5)
    labels = [
        ("format_and_parsability", "Format"),
        ("keyword_optimization", "Keywords"),
        ("experience_and_impact", "Experience"),
        ("skills_alignment", "Skills"),
        ("clarity_and_readability", "Clarity"),
    ]
    for column, (key, label) in zip(score_cols, labels):
        value = int(section_scores.get(key, 0))
        value = max(0, min(100, value))
        column.metric(label, f"{value}/100")

    st.subheader("Strengths")
    for item in result.get("strengths", []):
        st.success(str(item))

    st.subheader("Improvements")
    for item in result.get("improvements", []):
        priority = item.get("priority", "Medium")
        issue = item.get("issue", "")
        recommendation = item.get("recommendation", "")
        st.markdown(f"**{priority}: {issue}**")
        st.write(recommendation)

    missing = result.get("missing_keywords", [])
    st.subheader("Suggested keywords")
    if missing:
        st.write(", ".join(str(x) for x in missing))
    else:
        st.write("No major missing keywords were identified.")

    st.subheader("Suggested rewrites")
    for item in result.get("rewrites", []):
        st.markdown(f"**Before:** {item.get('before', '')}")
        st.markdown(f"**After:** {item.get('after', '')}")
        st.divider()

    st.subheader("ATS checklist")
    checklist = result.get("ats_checklist", {})
    for key, value in checklist.items():
        label = key.replace("_", " ").title()
        st.checkbox(label, value=bool(value), disabled=True)


st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload your resume to get an estimated ATS score, keyword feedback, "
    "and practical improvements powered by Gemini Flash."
)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Gemini model: `{MODEL_NAME}`")
    st.info(
        "Tip: Add the target job description for more useful keyword matching."
    )

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT.",
)

job_description = st.text_area(
    "Target job description (optional)",
    height=220,
    placeholder="Paste the job description here for job-specific ATS analysis...",
)

if uploaded_file:
    st.caption(f"Selected: {uploaded_file.name}")

    if st.button("Analyze Resume", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting resume text and analyzing with Gemini..."):
                resume_text = extract_text(uploaded_file)

                if len(resume_text.strip()) < 100:
                    st.error(
                        "Very little text could be extracted. If this is a scanned/image-only PDF, "
                        "please use a text-based PDF or DOCX version."
                    )
                    st.stop()

                result = analyze_resume(resume_text, job_description)
                st.session_state["analysis_result"] = result

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

if "analysis_result" in st.session_state:
    st.divider()
    show_results(st.session_state["analysis_result"])

st.caption(
    "Privacy note: the extracted resume text is sent to Gemini for analysis. "
    "Do not upload highly sensitive information unless you are comfortable sharing it with the API provider."
)
