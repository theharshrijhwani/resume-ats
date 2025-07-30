import io
import json
import ollama
import pandas as pd
import streamlit as st
import plotly.express as px
from pypdf import PdfReader

MODEL_NAME = "llama3"

def generate_response(job_description, resume, prompt):
    final_prompt = resume + job_description + prompt
    response = ollama.generate(model=MODEL_NAME, prompt=final_prompt, stream=True)

    def write_stream(response):
        for chunk in response:
            yield chunk['response']

    st.write_stream(write_stream(response))

def handle_upload(uploaded_file):
    if uploaded_file is not None:
        pdf = PdfReader(io.BytesIO(uploaded_file.read()))
        resume = ""
        for page in pdf.pages:
            resume += page.extract_text()
        
        return resume
    else:
        raise FileNotFoundError("file not found")


st.set_page_config(page_title="ATS RESUME XPERT")
st.header("ATS TRACKING SYSTEM")

job_description = st.text_area("Write the job description", key="input")

uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])

if uploaded_file is not None:
    st.write("file uploaded successfully!")

submit1 = st.button("Tell me about my resume")
submit2 = st.button("How can I learn more or which skills shall I improve?")
submit3 = st.button("Percentage match")
submit4 = st.button("Heatmap")

input_prompt_1 = """
You are an expert career advisor and resume analyst.

The user has uploaded their resume. Your task is to:
- Read and interpret the resume content.
- Summarize the candidate’s key strengths, skills, qualifications, experience, and achievements.
- Identify the type of roles this resume is best suited for.
- Highlight any standout features like leadership, technical expertise, domain knowledge, or certifications.

Provide your response in well-organized sections like:
1. Summary
2. Key Skills
3. Suitable Job Roles
4. Notable Highlights

LIMIT RESPONSE TO 200 WORDS
"""

input_prompt_2 = """
You are an expert career coach and job market analyst.

Based on the user’s resume:
- Identify missing or underrepresented skills relevant to today’s job market.
- Suggest both technical and soft skills that the user should learn to improve job-readiness.
- Recommend learning resources or certifications if possible (optional).
- Tailor your recommendations based on the career level (e.g., student, junior, mid-level, etc.).

Organize your response in:
1. Skill Gaps
2. Recommended Skills to Learn
3. Learning Suggestions

LIMIT RESPONSE TO 200 WORDS
"""

input_prompt_3 = """
You are a job description and resume matching engine.

Compare the user's resume with the provided job description (if available).
- Calculate an overall match percentage between resume and job role.
- Evaluate the alignment across these categories: Skills, Experience, Education, and Keywords.
- For each category, give a percentage match and explain why.
- If any section is weak or missing, provide brief improvement tips.

Respond in this structure:
1. Overall Match Percentage: XX%
2. Category-wise Breakdown:
   - Skills: XX% — explanation
   - Experience: XX% — explanation
   - Education: XX% — explanation
   - Keywords Match: XX% — explanation
3. Suggestions to Improve Match

LIMIT RESPONSE TO 200 WORDS
"""

input_prompt_4 = """
You are an ATS Assistant evaluating a resume (shown in the image) based on the provided job description.

Your task is to evaluate the relevance of the following **four sections** in the resume:
1. Skills
2. Experience
3. Projects
4. Education

Each section should be given a relevance score between 0 and 100 based on how well it matches the job description.

### OUTPUT FORMAT (STRICT):

```json
[
  {"section": "Skills", "score": <score>, "explanation": "the reason for score"},
  {"section": "Experience", "score": <score>, "explanation": "the reason for score"},
  {"section": "Projects", "score": <score>, "explanation": "the reason for score"},
  {"section": "Education", "score": <score>, "explanation": "the reason for score"}
]

THE OUTPUT SHOULD ONLY BE THE JSON IN BELOW FORMAT. NO ADDITIONAL LINES APART FROM THAT. DO NOT INCLUDE ANY EXPLANATION OR COMMENTARY:

"""

if submit1:
    if uploaded_file is not None:
        resume = handle_upload(uploaded_file)
        st.subheader("RESPONSE")
        response = generate_response(job_description, resume, input_prompt_1)
        # st.write(response)
    else: 
        st.write("please upload your resume")

if submit2:
    if uploaded_file is not None:
        resume = handle_upload(uploaded_file)
        st.subheader("RESPONSE")
        response = generate_response(job_description, resume, input_prompt_2)
        # st.write(response)
    else: 
        st.write("please upload your resume")

if submit3:
    if uploaded_file is not None:
        resume = handle_upload(uploaded_file)
        st.subheader("RESPONSE")
        response = generate_response(job_description, resume, input_prompt_3)
        # st.write(response)
    else: 
        st.write("please upload your resume")

if submit4:
    if uploaded_file is not None:
        resume = handle_upload(uploaded_file)
        final_prompt = resume + job_description + input_prompt_4
        response = ollama.generate(prompt=final_prompt, model="llama3", stream=False)
        parsed_data = json.loads(response["response"])
        df = pd.DataFrame(parsed_data)
        print('checkpoint')
        fig = px.imshow(
            df[['score']].T,
            labels=dict(x="Section", y="", color="Relevance Score"),
            x=df['section'],
            y=["Score"],
            color_continuous_scale='RdYlGn',
            text_auto=True
        )
        st.subheader("Resume Section Relevance Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Section-wise Explanations"):
            for row in parsed_data:
                st.markdown(f"**{row['section']}** — {row['score']}%")
                st.write(row['explanation'])
                st.markdown("---")
