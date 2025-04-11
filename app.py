import streamlit as st
import time
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random
import pandas as pd
import io
import base64
import re

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text() or ""  # Handle None case
            text += extracted_text
        
        if not text.strip():
            # If no text was extracted (possibly an image-based PDF)
            return "This appears to be an image-based PDF. Please provide a text-based PDF or manually enter the content."
            
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to read job description from CSV
import pandas as pd

def read_jd_from_csv(uploaded_file): 
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            return "The CSV file is empty."

        # Look for job description column
        jd_columns = ['job_description', 'jobdescription', 'description', 'jd', 'job', 'text', 'Job Description']
        jd_column = next((col for col in jd_columns if col in df.columns), None)

        if jd_column is None:
            return "No recognizable job description column found."

        # Look for job title column
        title_columns = ['job_title', 'title', 'position', 'Job Title']
        title_column = next((col for col in title_columns if col in df.columns), None)

        if title_column is None:
            return "No recognizable job title column found."

        # Extract and format
        formatted_descriptions = []
        for i, row in df[[title_column, jd_column]].dropna().iterrows():
            title = str(row[title_column]).strip()
            desc = str(row[jd_column]).strip()
            formatted_descriptions.append(f"---\n**{title}:**\n{desc}\n")

        return "\n".join(formatted_descriptions)

    except Exception as e:
        return f"Error reading CSV file: {str(e)}"



# Job Description Summarizer Agent
def summarize_job_description(jd_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as an expert job description analyzer. Review the following job description and extract 
    key elements in a structured format.
    
    Job Description: {jd_text}
    
    Respond with a valid JSON object containing these fields:
    {{
      "JobTitle": "title here",
      "Department": "department name",
      "Location": "location",
      "EmploymentType": "full-time/part-time/contract",
      "RequiredSkills": ["skill1", "skill2", "..."],
      "RequiredExperience": "X years in...",
      "RequiredQualifications": ["qualification1", "qualification2", "..."],
      "Responsibilities": ["responsibility1", "responsibility2", "..."],
      "SalaryRange": "range if mentioned",
      "PreferredSkills": ["skill1", "skill2", "..."]
    }}
    
    Important: Only respond with the JSON object and nothing else. No explanations or markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains markdown or non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                # Look for JSON object in the response
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Try to parse as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # If direct parsing fails, try a fallback approach
                return {
                    "JobTitle": extract_field_from_text(response_text, "JobTitle") or "Data Analyst",
                    "Department": extract_field_from_text(response_text, "Department") or "Not specified",
                    "Location": extract_field_from_text(response_text, "Location") or "Not specified",
                    "EmploymentType": extract_field_from_text(response_text, "EmploymentType") or "Full-time",
                    "RequiredSkills": extract_list_from_text(response_text, "RequiredSkills") or ["Python", "Data Analysis"],
                    "RequiredExperience": extract_field_from_text(response_text, "RequiredExperience") or "2+ years",
                    "RequiredQualifications": extract_list_from_text(response_text, "RequiredQualifications") or ["Bachelor's degree"],
                    "Responsibilities": extract_list_from_text(response_text, "Responsibilities") or ["Data Analysis", "Reporting"],
                    "SalaryRange": extract_field_from_text(response_text, "SalaryRange") or "Not specified",
                    "PreferredSkills": extract_list_from_text(response_text, "PreferredSkills") or []
                }
        else:
            return {"error": "Failed to get a valid response from the API"}
            
    except Exception as e:
        return {"error": f"Failed to process the JD: {str(e)}"}

# Helper functions to extract info from text if JSON parsing fails
def extract_field_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": "value"
    import re
    pattern = f'"{field_name}"\\s*:\\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_list_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": ["value1", "value2"]
    import re
    pattern = f'"{field_name}"\\s*:\\s*\\[(.*?)\\]'
    match = re.search(pattern, text)
    if match:
        items_text = match.group(1)
        # Extract individual items
        items = re.findall(r'"([^"]*)"', items_text)
        return items
    return None

# Recruiting Agent for CV Analysis
def analyze_cv(cv_text, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Format JD summary for the prompt
    required_skills = ", ".join(jd_summary.get("RequiredSkills", []))
    preferred_skills = ", ".join(jd_summary.get("PreferredSkills", []))
    responsibilities = ", ".join(jd_summary.get("Responsibilities", []))
    qualifications = ", ".join(jd_summary.get("RequiredQualifications", []))
    
    prompt = f"""
    Act as a senior recruiting agent specializing in talent acquisition. Analyze this candidate's 
    resume against the job requirements and provide a detailed evaluation.
    
    Job Title: {jd_summary.get("JobTitle", "Not specified")}
    Required Skills: {required_skills}
    Preferred Skills: {preferred_skills}
    Required Experience: {jd_summary.get("RequiredExperience", "Not specified")}
    Required Qualifications: {qualifications}
    Key Responsibilities: {responsibilities}
    
    Candidate Resume: {cv_text}
    
    Respond with ONLY a valid JSON object containing:
    {{
      "CandidateName": "full name",
      "ContactInfo": "email and/or phone",
      "Skills": ["skill1", "skill2", "..."],
      "Experience": ["experience1", "experience2", "..."],
      "Education": ["education1", "education2", "..."],
      "Certifications": ["cert1", "cert2", "..."],
      "SkillMatch": "X%",
      "ExperienceMatch": "X%",
      "QualificationMatch": "X%",
      "OverallMatch": "X%",
      "MatchedSkills": ["skill1", "skill2", "..."],
      "MissingSkills": ["skill1", "skill2", "..."],
      "Strengths": ["strength1", "strength2", "..."],
      "Areas_for_Improvement": ["area1", "area2", "..."],
      "Recommendation": "shortlist/reject/further review"
    }}
    
    Important: Only provide the JSON object. No additional text, no markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Create a fallback response if parsing fails
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Extract name from resume if possible
                candidate_name = "Unknown Candidate"
                name_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)", cv_text[:500])
                if name_match:
                    candidate_name = name_match.group(1)
                
                # Extract email if possible
                contact_info = "Not found"
                email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cv_text)
                if email_match:
                    contact_info = email_match.group(0)
                
                return {
                    "CandidateName": candidate_name,
                    "ContactInfo": contact_info,
                    "Skills": ["Unable to parse skills"],
                    "Experience": ["Experience details not parsed"],
                    "Education": ["Education details not parsed"],
                    "Certifications": [],
                    "SkillMatch": "0%",
                    "ExperienceMatch": "0%",
                    "QualificationMatch": "0%",
                    "OverallMatch": "50%",
                    "MatchedSkills": [],
                    "MissingSkills": jd_summary.get("RequiredSkills", []),
                    "Strengths": ["Unable to determine strengths"],
                    "Areas_for_Improvement": ["Resume parsing failed, please review manually"],
                    "Recommendation": "further review"
                }
        else:
            return {"error": "Failed to get a valid response from the API for CV analysis"}
            
    except Exception as e:
        return {"error": f"Failed to analyze CV: {str(e)}"}

# Candidate Shortlisting Agent
def shortlist_candidates(candidates_analysis, threshold=70):
    shortlisted = []
    
    for candidate in candidates_analysis:
        # Skip entries with errors
        if "error" in candidate:
            continue
            
        # Extract match percentage
        match_percentage = int(candidate.get("OverallMatch", "0%").strip("%"))
        
        if match_percentage >= threshold:
            shortlisted.append({
                "name": candidate.get("CandidateName", "Unknown"),
                "contact": candidate.get("ContactInfo", "Not provided"),
                "match_percentage": match_percentage,
                "strengths": candidate.get("Strengths", []),
                "missing_skills": candidate.get("MissingSkills", []),
                "recommendation": candidate.get("Recommendation", "")
            })
    
    # Sort by match percentage (highest first)
    shortlisted.sort(key=lambda x: x["match_percentage"], reverse=True)
    return shortlisted

# Interview Scheduler Agent
def generate_interview_email(candidate_info, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate interview dates (next business days)
    today = datetime.now()
    proposed_dates = []
    
    for i in range(1, 8):
        next_date = today + timedelta(days=i)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if next_date.weekday() < 5:
            proposed_dates.append(next_date.strftime("%A, %B %d, %Y"))
            if len(proposed_dates) == 3:  # Get 3 business days
                break
    
    # Generate interview times
    interview_times = ["10:00 AM", "11:30 AM", "2:00 PM", "3:30 PM"]
    proposed_slots = [f"{date} at {time}" for date in proposed_dates for time in random.sample(interview_times, 2)]
    
    job_title = jd_summary.get("JobTitle", "the open position")
    company = os.getenv("COMPANY_NAME", "Our Company")
    
    # Handle missing strengths
    candidate_strengths = candidate_info.get('strengths', [])
    if not candidate_strengths or len(candidate_strengths) == 0:
        candidate_strengths = ["qualifications", "experience"]
    
    strengths_text = ', '.join(candidate_strengths[:3]) if len(candidate_strengths) > 0 else "qualifications"
    
    prompt = f"""
    Act as a professional recruiter. Write a personalized interview invitation email for {candidate_info['name']} 
    who has been shortlisted for the {job_title} position at {company}.
    
    Candidate's strengths: {strengths_text}
    Match rate: {candidate_info['match_percentage']}%
    
    Include these proposed interview slots:
    {', '.join(proposed_slots[:5])}
    
    The email should:
    1. Be professional and warm
    2. Congratulate them on being shortlisted
    3. Briefly mention why they're a good fit, highlighting 1-2 strengths
    4. Propose the interview slots and ask for their preference
    5. Mention the interview will be conducted via video call (Zoom)
    6. Explain next steps and whom to contact with questions
    
    Respond with only the email text, no additional formatting or explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            email_text = response.text.strip()
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": email_text,
                "proposed_slots": proposed_slots[:5]
            }
        else:
            # Fallback email if API fails
            default_email = f"""
Dear {candidate_info['name']},

Congratulations! We are pleased to inform you that you have been shortlisted for the {job_title} position at {company}.

We were impressed with your profile and would like to invite you for a video interview to discuss your experience and the role in more detail.

Please let us know which of the following time slots would work best for you:
- {proposed_slots[0]}
- {proposed_slots[1]}
- {proposed_slots[2]}

The interview will be conducted via Zoom, and we will send you the meeting details once you confirm your preferred time slot.

If you have any questions, please don't hesitate to contact us.

We look forward to speaking with you soon!

Best regards,
Recruiting Team
{company}
            """
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": default_email,
                "proposed_slots": proposed_slots[:5]
            }
    except Exception as e:
        return {"error": f"Failed to generate email: {str(e)}"}

# Function to create a mailto link for email
def generate_mailto_link(email_data):
    try:
        recipient = email_data['candidate_email']
        subject = email_data['email_subject']
        body = email_data['email_body']
        
        # URL encode the subject and body for the mailto link
        import urllib.parse
        subject_encoded = urllib.parse.quote(subject)
        body_encoded = urllib.parse.quote(body)
        
        # Create the mailto link
        mailto_link = f"mailto:{recipient}?subject={subject_encoded}&body={body_encoded}"
        
        return {
            "status": "success",
            "mailto_link": mailto_link,
            "message": f"Email ready to send to {recipient}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating email link: {str(e)}"
        }

# Function to get image as base64 for embedded display
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Page configuration
st.set_page_config(
    page_title="EzHunt | Smart Recruitment Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #5B6EF7;
        --primary-light: #DCE1FF;
        --primary-dark: #3A4CCA;
        --accent-color: #FF6B6B;
        --success-color: #63D471;
        --warning-color: #FFDA83;
        --danger-color: #F45B69;
        --text-color: #333F50;
        --light-text: #7E8CA0;
        --background-color: #F7F9FC;
        --card-color: #FFFFFF;
        --border-radius: 12px;
        --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Base styles */
    body {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--box-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='rgba(255,255,255,0.1)' fill-rule='evenodd'/%3E%3C/svg%3E");
        opacity: 0.6;
    }
    
    /* Logo and brand */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .logo-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-left: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background-color: var(--card-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--box-shadow);
        margin-bottom: 1.5rem;
        border-top: 4px solid var(--primary-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Step cards with indicators */
    .step-card {
        border-left: 5px solid var(--primary-color);
        background-color: white;
        border-radius: var(--border-radius);
        padding: 1.25rem;
        box-shadow: var(--box-shadow);
        margin-bottom: 1.25rem;
        transition: all 0.3s ease;
    }
    
    .step-complete {
        border-left: 5px solid var(--success-color);
    }
    
    .step-active {
        border-left: 5px solid var(--warning-color);
        background-color: #FFFDF7;
    }
    
    .step-waiting {
        border-left: 5px solid #E0E0E0;
        opacity: 0.7;
    }
    
    /* Progress indicator */
    .progress-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .progress-container::before {
        content: "";
        position: absolute;
        top: 15px;
        left: 0;
        width: 100%;
        height: 4px;
        background-color: #E0E0E0;
        z-index: 1;
    }
    
    .progress-step {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background-color: white;
        border: 3px solid #E0E0E0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        z-index: 2;
        position: relative;
    }
    
    .progress-step.active {
        border-color: var(--primary-color);
        background-color: var(--primary-color);
        color: white;
    }
    
    .progress-step.completed {
        border-color: var(--success-color);
        background-color: var(--success-color);
        color: white;
    }
    
    .progress-step-label {
        position: absolute;
        top: 40px;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--light-text);
    }
    
    .progress-step.active .progress-step-label {
        color: var(--primary-color);
    }
    
    .progress-step.completed .progress-step-label {
        color: var(--success-color);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 4px 12px rgba(91, 110, 247, 0.3);
        transform: translateY(-2px);
    }
    
    .secondary-button {
        background-color: #F0F2F5 !important;
        color: var(--text-color) !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    .secondary-button:hover {
        background-color: #E5E8EC !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    .action-button {
        background-color: var(--success-color) !important;
    }
    
    .action-button:hover {
        background-color: #4FBD5D !important;
        box-shadow: 0 4px 12px rgba(99, 212, 113, 0.3) !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        margin: 0.25rem;
        background-color: var(--primary-light);
        color: var(--primary-dark);
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Matched keyword pills */
    .matched-keyword {
        background-color: rgba(99, 212, 113, 0.2);
        color: #2a6133;
    }
    
    /* Missing keyword pills */
    .missing-keyword {
        background-color: rgba(244, 91, 105, 0.2);
        color: #8b2833;
    }
    
    /* Match progress bar */
    .match-progress {
        height: 8px;
        border-radius: 4px;
        background-color: #EEF0F5;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    .match-progress-bar {
        height: 100%;
        transition: width 0.8s ease;
    }
    
    /* Match percentage colors */
    .match-high {
        background-color: var(--success-color);
    }
    
    .match-medium {
        background-color: var(--warning-color);
    }
    
    .match-low {
        background-color: var(--danger-color);
    }
    
    /* Candidate card */
    .candidate-card {
        background-color: white;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border-left: 4px solid #E0E0E0;
    }
    
    .candidate-card.selected {
        border-left: 4px solid var(--success-color);
        background-color: rgba(99, 212, 113, 0.05);
            border-left: 4px solid var(--success-color);
        background-color: rgba(99, 212, 113, 0.05);
        box-shadow: 0 4px 12px rgba(99, 212, 113, 0.2);
    }
    
    .candidate-card.shortlisted {
        border-left: 4px solid var(--primary-color);
    }
    
    .candidate-card.rejected {
        border-left: 4px solid var(--danger-color);
        opacity: 0.7;
    }
    
    /* Email preview */
    .email-preview {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid #E0E0E0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Dashboard stats */
    .stat-card {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--box-shadow);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: var(--light-text);
        font-size: 0.9rem;
    }
    
    /* Notifications */
    .notification {
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        border-radius: var(--border-radius);
        background-color: var(--primary-light);
        border-left: 4px solid var(--primary-color);
        font-size: 0.9rem;
    }
    
    .notification.success {
        background-color: rgba(99, 212, 113, 0.1);
        border-left: 4px solid var(--success-color);
    }
    
    .notification.warning {
        background-color: rgba(255, 218, 131, 0.1);
        border-left: 4px solid var(--warning-color);
    }
    
    .notification.error {
        background-color: rgba(244, 91, 105, 0.1);
        border-left: 4px solid var(--danger-color);
    }
    
    /* Streamlit specific adjustments */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: var(--card-color);
        padding: 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Remove default backgrounds and padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }

    /* Fix sidebar background */
    [data-testid="stSidebar"] {
        background-color: var(--background-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom file uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #E0E0E0;
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
        background-color: rgba(91, 110, 247, 0.05);
    }
    
    /* Custom metric styling */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--box-shadow);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--light-text) !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if 'jd_summary' not in st.session_state:
    st.session_state.jd_summary = None
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'shortlisted' not in st.session_state:
    st.session_state.shortlisted = []
if 'selected_candidate' not in st.session_state:
    st.session_state.selected_candidate = None
if 'interview_email' not in st.session_state:
    st.session_state.interview_email = None

# Functions to navigate steps
def go_to_step(step):
    st.session_state.current_step = step

def next_step():
    st.session_state.current_step += 1

def prev_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1

# Header
st.markdown('<div class="main-header"><div class="logo-container">🎯 <span class="logo-text">EzHunt</div><p>AI-Powered Recruitment Assistant</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    
    # Create progress steps
    steps = [
        "Import", 
        "Upload", 
        "Review", 
        "Shortlist", 
        "Schedule"
    ]
    
    # Display progress indicator
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    
    for i, step in enumerate(steps, 1):
        with cols[i-1]:
            if i < st.session_state.current_step:
                # Completed step
                st.markdown(f"""
                <div class="progress-step completed">✓
                    <div class="progress-step-label">{step}</div>
                </div>
                """, unsafe_allow_html=True)
            elif i == st.session_state.current_step:
                # Current step
                st.markdown(f"""
                <div class="progress-step active">{i}
                    <div class="progress-step-label">{step}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Future step
                st.markdown(f"""
                <div class="progress-step">{i}
                    <div class="progress-step-label">{step}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Candidates</div>
            <div class="stat-number">{len(st.session_state.candidates)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Shortlisted</div>
            <div class="stat-number">{len(st.session_state.shortlisted)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tips
    st.markdown("### Tips")
    tips = [
        "📄 PDF resumes work best for analysis",
        "🔍 More detailed job descriptions yield better matches",
        "⚖️ Consider both skills and experience when reviewing",
        "🔄 You can return to previous steps anytime"
    ]
    
    for tip in tips:
        st.markdown(f"""
        <div class="notification">
            {tip}
        </div>
        """, unsafe_allow_html=True)
    
    # About & Help
    st.markdown("### About")
    st.markdown("""
    EzHunt is an AI-driven recruitment assistant that automates job screening by parsing job descriptions, analyzing resumes, computing match scores, and shortlisting top candidates.
    
    Need help? Contact support@ezhunt.com [Eg.]
    """)

# Main content area based on current step
if st.session_state.current_step == 1:
    st.markdown("## Step 1: Import Job Description")
    
    tab1, tab2, tab3 = st.tabs(["Upload File", "Paste Text", "Example"])
    
    with tab1:
        st.markdown("Upload your job description file (PDF or CSV format)")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv"], key="jd_file")
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            # Extract text based on file type
            if uploaded_file.name.endswith('.pdf'):
                st.session_state.jd_text = input_pdf_text(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                st.session_state.jd_text = read_jd_from_csv(uploaded_file)
            
            # Show preview
            with st.expander("Preview Job Description", expanded=True):
                st.text_area("Job Description Text", st.session_state.jd_text, height=200)
    
    with tab2:
        st.markdown("Paste your job description text directly")
        
        jd_input = st.text_area("Job Description", height=300, key="jd_input",
                               placeholder="Paste the full job description here...")
        
        if st.button("Use This Text"):
            if jd_input.strip():
                st.session_state.jd_text = jd_input
                st.success("Job description saved!")
            else:
                st.error("Please enter job description text")
    
    with tab3:
        st.markdown("Use an example job description to test the system")
        
        example_jds = {
            "Data Analyst": """
            Job Title: Data Analyst
            Department: Business Intelligence
            Location: Chicago, IL (Hybrid)
            
            About Us:
            Our company is a leading provider of data-driven solutions for the healthcare industry. We are seeking a skilled Data Analyst to join our growing team.
            
            Role Overview:
            As a Data Analyst, you will be responsible for analyzing complex datasets, creating visualizations, and providing insights to support business decision-making. You will work closely with cross-functional teams to understand their data needs and deliver actionable intelligence.
            
            Responsibilities:
            • Collect, process, and analyze large datasets from multiple sources
            • Develop and maintain dashboards and reports using tools like Tableau or Power BI
            • Identify trends, patterns, and anomalies in data
            • Collaborate with stakeholders to understand their data needs
            • Present findings to technical and non-technical audiences
            • Support data-driven decision making across the organization
            • Ensure data quality and integrity
            
            Required Qualifications:
            • Bachelor's degree in Statistics, Mathematics, Computer Science, or related field
            • 2+ years of experience in data analysis or business intelligence
            • Proficiency in SQL and database querying
            • Experience with data visualization tools (Tableau, Power BI)
            • Strong skills in Excel and data manipulation
            • Excellent analytical and problem-solving abilities
            
            Required Skills:
            • SQL
            • Python or R
            • Data Visualization
            • Statistical Analysis
            • Excel (Advanced)
            
            Preferred Skills:
            • Healthcare industry experience
            • Knowledge of machine learning concepts
            • Experience with cloud platforms (AWS, Azure)
            • ETL processes and data warehousing
            
            Salary Range: $70,000 - $90,000 DOE
            
            Employment Type: Full-time
            
            Benefits:
            • Comprehensive health, dental, and vision insurance
            • 401(k) matching
            • Flexible work arrangements
            • Professional development opportunities
            • Paid time off and holidays
            """,
            
            "Frontend Developer": """
            Job Title: Frontend Developer
            Department: Engineering
            Location: Remote (US-based)
            
            About the Role:
            We are looking for a skilled Frontend Developer to join our engineering team. As a Frontend Developer, you will be responsible for implementing visual elements and user interactions that users engage with through their web browser or application.
            
            Key Responsibilities:
            • Develop new user-facing features using React.js
            • Build reusable components and front-end libraries for future use
            • Optimize components for maximum performance across devices and browsers
            • Translate designs and wireframes into high-quality code
            • Collaborate with backend developers to integrate UI components with APIs
            • Implement responsive design and ensure cross-browser compatibility
            
            Required Skills & Experience:
            • 3+ years of experience in frontend development
            • Strong proficiency in JavaScript, including DOM manipulation and ES6+
            • Thorough understanding of React.js and its core principles
            • Experience with common frontend development tools like Webpack, Babel, etc.
            • Familiarity with RESTful APIs and modern frontend build pipelines
            • Excellent HTML5 and CSS3 skills including preprocessors like SASS/LESS
            • Understanding of server-side rendering and its benefits/drawbacks
            
            Required Qualifications:
            • Bachelor's degree in Computer Science or related field (or equivalent experience)
            • Portfolio of web applications or examples of released code
            
            Preferred Skills:
            • Experience with TypeScript
            • Knowledge of modern UI/UX design principles
            • Experience with state management libraries (Redux, MobX)
            • Understanding of CI/CD pipelines
            • Experience with testing frameworks (Jest, Enzyme, etc.)
            
            Employment Type: Full-time
            
            Salary Range: $90,000 - $120,000 depending on experience
            
            Benefits:
            • Health, dental, and vision insurance
            • 401(k) with company match
            • Unlimited PTO policy
            • Home office stipend
            • Professional development budget
            """,
            
            "Project Manager": """
            Job Title: Project Manager
            Department: Operations
            Location: Boston, MA
            
            Company Overview:
            Our company specializes in delivering enterprise software solutions to the financial services industry. We're looking for an experienced Project Manager to join our team and lead projects from inception to completion.
            
            Job Description:
            The Project Manager will be responsible for planning, executing, and finalizing projects according to strict deadlines and within budget. This includes acquiring resources and coordinating the efforts of team members and third-party contractors or consultants in order to deliver projects according to plan.
            
            Key Responsibilities:
            • Define project scope, goals, and deliverables in collaboration with senior management
            • Develop full-scale project plans and associated communications documents
            • Manage project budget and resource allocation
            • Track project milestones and deliverables using appropriate tools
            • Coordinate internal resources and third parties/vendors for the flawless execution of projects
            • Ensure that all projects are delivered on-time, within scope and within budget
            • Develop and maintain comprehensive project documentation
            • Perform risk assessments and develop mitigation strategies
            • Report project status to management and stakeholders
            
            Requirements:
            • Bachelor's degree in Business Administration, Computer Science, or related field
            • 5+ years of project management experience, preferably in software development or IT
            • PMP certification strongly preferred
            • Proven experience in managing complex projects from concept to completion
            • Strong knowledge of project management methodologies (Agile, Scrum, Waterfall)
            • Proficiency with project management tools (JIRA, MS Project, Asana, etc.)
            • Excellent communication and leadership skills
            • Strong problem-solving abilities and analytical skills
            
            Required Skills:
            • Project Planning
            • Budget Management
            • Risk Management
            • Stakeholder Communication
            • Team Leadership
            • Agile Methodologies
            
            Preferred Skills:
            • Knowledge of financial services industry
            • Experience with enterprise software implementation
            • Change management experience
            • Scrum Master certification
            
            Employment Type: Full-time
            Salary Range: $95,000 - $120,000 based on experience
            """
        }
        
        selected_example = st.selectbox("Select an example job description", list(example_jds.keys()))
        
        if st.button("Use This Example"):
            st.session_state.jd_text = example_jds[selected_example]
            st.success(f"Example job description for {selected_example} loaded!")
            
            # Show preview
            with st.expander("Preview Job Description", expanded=True):
                st.text_area("Job Description Text", st.session_state.jd_text, height=200)
    
    # Process Job Description
    if st.session_state.jd_text:
        if st.button("Analyze Job Description", type="primary"):
            with st.spinner("Analyzing job description..."):
                # Display progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Analyze the JD
                st.session_state.jd_summary = summarize_job_description(st.session_state.jd_text)
                
                # Show results
                if "error" not in st.session_state.jd_summary:
                    st.success("Job description analyzed successfully!")
                    next_step()
                    st.experimental_rerun()
                else:
                    st.error(f"Error analyzing job description: {st.session_state.jd_summary['error']}")
    else:
        st.info("Please import or enter a job description to continue")

elif st.session_state.current_step == 2:
    st.markdown("## Step 2: Upload Resumes")
    
    # Display JD summary
    if st.session_state.jd_summary:
        with st.expander("Job Description Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Job Title:** {st.session_state.jd_summary.get('JobTitle', 'Not specified')}")
                st.markdown(f"**Department:** {st.session_state.jd_summary.get('Department', 'Not specified')}")
                st.markdown(f"**Location:** {st.session_state.jd_summary.get('Location', 'Not specified')}")
                st.markdown(f"**Employment Type:** {st.session_state.jd_summary.get('EmploymentType', 'Not specified')}")
                
            with col2:
                st.markdown(f"**Experience Required:** {st.session_state.jd_summary.get('RequiredExperience', 'Not specified')}")
                st.markdown(f"**Salary Range:** {st.session_state.jd_summary.get('SalaryRange', 'Not specified')}")
            
            st.markdown("### Required Skills")
            skill_html = ""
            for skill in st.session_state.jd_summary.get('RequiredSkills', []):
                skill_html += f'<span class="keyword-pill">{skill}</span>'
            
            st.markdown(f"<div>{skill_html}</div>", unsafe_allow_html=True)
            
            st.markdown("### Preferred Skills")
            pref_skill_html = ""
            for skill in st.session_state.jd_summary.get('PreferredSkills', []):
                pref_skill_html += f'<span class="keyword-pill">{skill}</span>'
            
            if not pref_skill_html:
                pref_skill_html = "<em>None specified</em>"
                
            st.markdown(f"<div>{pref_skill_html}</div>", unsafe_allow_html=True)
            
            st.markdown("### Key Responsibilities")
            for resp in st.session_state.jd_summary.get('Responsibilities', [])[:5]:
                st.markdown(f"- {resp}")
    
    # Upload resumes
    st.markdown("### Upload Candidate Resumes")
    
    tab1, tab2 = st.tabs(["Upload Files", "Example Candidates"])
    
    with tab1:
        st.markdown("Upload one or more candidate resumes (PDF format)")
        
        uploaded_files = st.file_uploader("Choose files", type=["pdf"], accept_multiple_files=True, key="cv_files")
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            
            # Store the files for processing
            if st.button("Process Resumes", type="primary"):
                with st.spinner("Extracting text from resumes..."):
                    progress_bar = st.progress(0)
                    increment = 100 / (len(uploaded_files) * 2)  # Split progress between extraction and analysis
                    progress = 0
                    
                    # Extract text from each resume
                    cv_texts = []
                    for i, file in enumerate(uploaded_files):
                        cv_text = input_pdf_text(file)
                        cv_texts.append(cv_text)
                        progress += increment
                        progress_bar.progress(min(int(progress), 100))
                    
                    # Analyze each resume
                    analysis_results = []
                    for i, cv_text in enumerate(cv_texts):
                        result = analyze_cv(cv_text, st.session_state.jd_summary)
                        analysis_results.append(result)
                        progress += increment
                        progress_bar.progress(min(int(progress), 100))
                    
                    # Store results in session state
                    st.session_state.candidates = cv_texts
                    st.session_state.analysis_results = analysis_results
                    
                    # Show completion and move to next step
                    st.success("All resumes processed!")
                    next_step()
                    st.experimental_rerun()
    
    with tab2:
        st.markdown("Use example candidates to test the system")
        
        example_candidates = {
            "Strong Match": {
                "name": "Alex Johnson",
                "text": """
                Alex Johnson
                Data Analyst
                
                Contact Information:
                Email: alex.johnson@email.com
                Phone: (555) 123-4567
                LinkedIn: linkedin.com/in/alexjohnson
                
                Summary:
                Results-driven Data Analyst with 4+ years of experience analyzing complex datasets and creating visualizations to drive business decisions. Proficient in SQL, Python, and Tableau with a strong background in statistical analysis and data quality management. Known for translating complex data into actionable insights for both technical and non-technical stakeholders.
                
                Skills:
                • SQL (Advanced)
                • Python (pandas, NumPy, scikit-learn)
                • R for statistical analysis
                • Data Visualization (Tableau, Power BI)
                • Excel (Advanced: VLOOKUP, Pivot Tables, Macros)
                • Statistical Analysis
                • ETL Processes
                • Database Management
                • Data Cleaning and Validation
                • Dashboard Creation
                
                Experience:
                
                Senior Data Analyst | HealthTech Solutions | Chicago, IL | June 2021 - Present
                • Analyze patient outcome data for 15+ healthcare facilities, identifying trends that led to a 12% improvement in treatment protocols
                • Design and maintain executive dashboards in Tableau, providing real-time insights into key performance metrics
                • Develop SQL queries to extract and analyze data from multiple databases, enabling cross-functional analysis
                • Collaborate with product team to implement data-driven features, resulting in 20% increase in user engagement
                • Automate reporting processes using Python, reducing monthly reporting time by 40%
                
                Data Analyst | MarketWise Analytics | Chicago, IL | May 2019 - June 2021
                • Conducted A/B testing analysis for e-commerce clients, resulting in conversion rate improvements of up to 25%
                • Built predictive models to forecast customer behavior using Python's scikit-learn
                • Created comprehensive reports and visualizations that improved client decision-making processes
                • Managed a database of over 5 million customer records, ensuring data integrity and compliance
                • Developed and documented standard operating procedures for data analysis
                
                Business Intelligence Intern | DataCorp | Chicago, IL | Jan 2019 - May 2019
                • Assisted in the development of KPI dashboards using Power BI
                • Performed data cleansing and validation tasks to maintain data quality
                • Supported the analytics team in producing weekly and monthly reports
                
                Education:
                Bachelor of Science in Statistics | University of Illinois Chicago | 2018
                • Minor in Computer Science
                • Dean's List: 2016, 2017, 2018
                
                Certifications:
                • Microsoft Certified: Data Analyst Associate
                • Tableau Desktop Specialist
                • Google Analytics Certification
                
                Projects:
                Healthcare Outcomes Analysis Project
                • Analyzed 3 years of patient data to identify factors influencing treatment outcomes
                • Developed an interactive dashboard that allowed healthcare providers to visualize patient trends
                • Identified key demographic factors that correlated with treatment success rates
                
                E-commerce Customer Segmentation
                • Used K-means clustering to segment 50,000+ customers based on purchasing behavior
                • Created targeted marketing strategies for each segment, increasing sales by 15%
                • Implemented a Python-based recommendation engine that boosted cross-selling
                """
            },
            "Medium Match": {
                "name": "Jordan Smith",
                "text": """
                Jordan Smith
                Business Analyst
                
                Contact:
                jordan.smith@email.com
                (555) 987-6543
                Chicago, IL
                
                Professional Summary:
                Business Analyst with 3 years of experience working with data to drive business decisions. Experience in SQL, data visualization, and financial analysis. Looking to leverage my analytical skills in a data-focused role.
                
                Skills:
                • SQL (Intermediate)
                • Excel (Advanced)
                • Data Visualization (Tableau)
                • Business Intelligence
                • Financial Reporting
                • Process Improvement
                • Project Management
                • Statistical Analysis (Basic)
                
                Work Experience:
                
                Business Analyst | FinTech Solutions | Chicago, IL | Aug 2020 - Present
                • Analyze financial data and create monthly reports for executive leadership
                • Develop and maintain Tableau dashboards tracking key performance indicators
                • Collaborate with cross-functional teams to implement data-driven process improvements
                • Write SQL queries to extract data from various databases
                • Identify cost-saving opportunities through data analysis, resulting in 15% reduction in operational expenses
                
                Financial Analyst | Global Retail Corp | Chicago, IL | Jun 2018 - Aug 2020
                • Performed budget variance analysis and prepared monthly financial reports
                • Developed Excel models for sales forecasting and inventory management
                • Supported strategic initiatives through financial analysis and recommendations
                • Participated in process improvement projects, increasing department efficiency by 20%
                
                Education:
                Bachelor of Business Administration | DePaul University | 2018
                • Concentration in Finance and Information Systems
                
                Certifications:
                • Tableau Desktop Qualified Associate
                • Microsoft Excel Expert
                
                Projects:
                Sales Performance Dashboard
                • Created an interactive Tableau dashboard to track sales performance across regions
                • Identified underperforming markets and provided recommendations for improvement
                
                Budget Optimization Analysis
                • Analyzed departmental spending patterns to identify inefficiencies
                • Recommendations led to 12% annual cost reduction
                """
            },
            "Low Match": {
                "name": "Taylor Wilson",
                "text": """
                Taylor Wilson
                Marketing Specialist
                
                Contact Information:
                Email: taylor.wilson@email.com
                Phone: (555) 456-7890
                Location: Chicago, IL
                
                Professional Summary:
                Creative and results-oriented Marketing Specialist with 5 years of experience developing and implementing successful marketing campaigns. Skilled in digital marketing, content creation, and social media management. Passionate about using data to inform marketing strategies and measure campaign effectiveness.
                
                Skills:
                • Digital Marketing
                • Content Strategy
                • Social Media Management
                • Email Marketing
                • SEO/SEM
                • Marketing Analytics
                • Adobe Creative Suite
                • CRM Systems (Salesforce)
                • Project Management
                • Brand Development
                
                Professional Experience:
                
                Senior Marketing Specialist | BrandGrowth Agency | Chicago, IL | Jan 2021 - Present
                • Manage marketing campaigns for 10+ clients across various industries
                • Increased client social media engagement by an average of 45% through strategic content planning
                • Develop and implement SEO strategies resulting in 30% increase in organic traffic
                • Create monthly performance reports analyzing marketing KPIs and ROI
                • Lead a team of 3 marketing coordinators in campaign execution
                
                Marketing Coordinator | TechStart Inc. | Chicago, IL | Mar 2018 - Dec 2020
                • Executed multi-channel marketing campaigns for SaaS products
                • Created content for blog posts, social media, and email newsletters
                • Assisted in website redesign project that improved conversion rates by
                • Conducted basic data analysis to track campaign performance
                • Collaborated with sales team to generate leads and nurture prospects
                
                Marketing Intern | Chicago Media Group | Chicago, IL | Sep 2017 - Mar 2018
                • Assisted with social media content creation and scheduling
                • Conducted competitor research and prepared market analysis reports
                • Supported event planning and execution for client product launches
                
                Education:
                Bachelor of Arts in Communication | Loyola University Chicago | 2017
                • Minor in Business Administration
                
                Certifications:
                • Google Analytics Certification
                • HubSpot Content Marketing Certification
                • Facebook Blueprint Certification
                
                Key Projects:
                E-commerce Rebranding Campaign
                • Led complete rebranding effort for online retailer
                • Developed new visual identity and messaging strategy
                • Resulted in 65% increase in brand recognition according to customer surveys
                
                B2B Lead Generation Program
                • Created targeted content marketing strategy for SaaS client
                • Implemented marketing automation workflows
                • Generated 40% increase in qualified leads within first quarter
                """
            }
        }
        
        num_examples = st.number_input("How many example candidates do you want to use?", min_value=1, max_value=3, value=2)
        
        if st.button("Use Example Candidates"):
            with st.spinner("Processing example candidates..."):
                # Generate progress bar
                progress_bar = st.progress(0)
                
                # Get the selected number of examples
                examples_list = list(example_candidates.values())[:num_examples]
                
                # Store example CV texts
                cv_texts = [example["text"] for example in examples_list]
                st.session_state.candidates = cv_texts
                
                # Analyze each resume
                analysis_results = []
                for i, cv_text in enumerate(cv_texts):
                    # Update progress
                    progress = int((i / len(cv_texts)) * 100)
                    progress_bar.progress(progress)
                    
                    # Analyze CV
                    result = analyze_cv(cv_text, st.session_state.jd_summary)
                    analysis_results.append(result)
                
                # Complete progress bar
                progress_bar.progress(100)
                
                # Store results and move to next step
                st.session_state.analysis_results = analysis_results
                st.success(f"{num_examples} example candidates processed!")
                next_step()
                st.experimental_rerun()
    
    # Navigation buttons
    col1, col2 = st.columns([1, 5])
    with col1:# Navigation buttons
        col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Back"):
            prev_step()
            st.experimental_rerun()

# Step 3: Review Analysis
elif st.session_state.current_step == 3:
    st.markdown("## Step 3: Review Analysis")
    
    if st.session_state.analysis_results:
        st.markdown("### Candidate Analysis Results")
        
        # Create tabs for each candidate
        candidate_tabs = []
        for i, result in enumerate(st.session_state.analysis_results):
            if "error" in result:
                tab_name = f"Candidate {i+1} (Error)"
            else:
                tab_name = result.get("CandidateName", f"Candidate {i+1}")
            candidate_tabs.append(tab_name)
        
        tabs = st.tabs(candidate_tabs)
        
        # Display analysis for each candidate
        for i, (tab, result) in enumerate(zip(tabs, st.session_state.analysis_results)):
            with tab:
                if "error" in result:
                    st.error(f"Error analyzing this resume: {result['error']}")
                    continue
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {result.get('CandidateName', 'Unnamed Candidate')}")
                    st.markdown(f"**Contact:** {result.get('ContactInfo', 'Not available')}")
                    
                    # Match percentages
                    st.markdown("### Match Scores")
                    
                    # Skills match
                    skill_match = int(result.get('SkillMatch', '0%').strip('%'))
                    st.markdown(f"**Skills Match:** {skill_match}%")
                    skill_class = "match-high" if skill_match >= 70 else ("match-medium" if skill_match >= 50 else "match-low")
                    st.markdown(f"""
                    <div class="match-progress">
                        <div class="match-progress-bar {skill_class}" style="width: {skill_match}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Experience match
                    exp_match = int(result.get('ExperienceMatch', '0%').strip('%'))
                    st.markdown(f"**Experience Match:** {exp_match}%")
                    exp_class = "match-high" if exp_match >= 70 else ("match-medium" if exp_match >= 50 else "match-low")
                    st.markdown(f"""
                    <div class="match-progress">
                        <div class="match-progress-bar {exp_class}" style="width: {exp_match}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Qualification match
                    qual_match = int(result.get('QualificationMatch', '0%').strip('%'))
                    st.markdown(f"**Qualification Match:** {qual_match}%")
                    qual_class = "match-high" if qual_match >= 70 else ("match-medium" if qual_match >= 50 else "match-low")
                    st.markdown(f"""
                    <div class="match-progress">
                        <div class="match-progress-bar {qual_class}" style="width: {qual_match}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Overall match
                    overall_match = int(result.get('OverallMatch', '0%').strip('%'))
                    st.markdown(f"**Overall Match:** {overall_match}%")
                    overall_class = "match-high" if overall_match >= 70 else ("match-medium" if overall_match >= 50 else "match-low")
                    st.markdown(f"""
                    <div class="match-progress">
                        <div class="match-progress-bar {overall_class}" style="width: {overall_match}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendation
                    recommendation = result.get('Recommendation', 'further review')
                    st.markdown(f"**Recommendation:** {recommendation.capitalize()}")
                
                with col2:
                    # Skills section
                    st.markdown("### Skills")
                    
                    # Matched skills
                    st.markdown("**Matched Skills:**")
                    matched_html = ""
                    for skill in result.get('MatchedSkills', []):
                        matched_html += f'<span class="keyword-pill matched-keyword">{skill}</span>'
                    
                    if not matched_html:
                        matched_html = "<em>No matched skills</em>"
                        
                    st.markdown(f"<div>{matched_html}</div>", unsafe_allow_html=True)
                    
                    # Missing skills
                    st.markdown("**Missing Skills:**")
                    missing_html = ""
                    for skill in result.get('MissingSkills', []):
                        missing_html += f'<span class="keyword-pill missing-keyword">{skill}</span>'
                    
                    if not missing_html:
                        missing_html = "<em>No missing skills</em>"
                        
                    st.markdown(f"<div>{missing_html}</div>", unsafe_allow_html=True)
                
                # Strengths and Areas for Improvement
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Strengths")
                    strengths = result.get('Strengths', [])
                    if strengths:
                        for strength in strengths:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("*No strengths identified*")
                
                with col2:
                    st.markdown("### Areas for Improvement")
                    improvements = result.get('Areas_for_Improvement', [])
                    if improvements:
                        for improvement in improvements:
                            st.markdown(f"- {improvement}")
                    else:
                        st.markdown("*No specific improvement areas identified*")
                
                # Experience and Education
                with st.expander("Experience & Education", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Experience:**")
                        experience = result.get('Experience', [])
                        if experience:
                            for exp in experience:
                                st.markdown(f"- {exp}")
                        else:
                            st.markdown("*No experience details extracted*")
                    
                    with col2:
                        st.markdown("**Education:**")
                        education = result.get('Education', [])
                        if education:
                            for edu in education:
                                st.markdown(f"- {edu}")
                        else:
                            st.markdown("*No education details extracted*")
                        
                        if 'Certifications' in result and result['Certifications']:
                            st.markdown("**Certifications:**")
                            for cert in result['Certifications']:
                                st.markdown(f"- {cert}")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 5])
        
        with col1:
            if st.button("⬅️ Back"):
                prev_step()
                st.experimental_rerun()
        
        with col2:
            if st.button("Next ➡️", type="primary"):
                next_step()
                st.experimental_rerun()
    else:
        st.warning("No candidate analysis results available. Please go back and process resumes first.")
        
        if st.button("⬅️ Back to Upload Resumes"):
            prev_step()
            st.experimental_rerun()

# Step 4: Shortlist Candidates
elif st.session_state.current_step == 4:
    st.markdown("## Step 4: Shortlist Candidates")
    
    if st.session_state.analysis_results:
        st.markdown("### Candidate Assessment")
        
        # Threshold for shortlisting
        threshold = st.slider("Shortlisting Threshold (Overall Match %)", min_value=50, max_value=90, value=70, step=5)
        
        if st.button("Generate Shortlist"):
            with st.spinner("Generating shortlist..."):
                # Generate shortlist based on threshold
                st.session_state.shortlisted = shortlist_candidates(st.session_state.analysis_results, threshold)
                st.success(f"Shortlist generated! {len(st.session_state.shortlisted)} candidates meet the {threshold}% threshold.")
        
        # Display all candidates with filtering options
        st.markdown("### All Candidates")
        
        # Sort options
        sort_option = st.selectbox("Sort by", ["Match % (High to Low)", "Match % (Low to High)", "Name (A-Z)"])
        
        # Create a list of candidates with their analysis
        candidates_list = []
        for i, result in enumerate(st.session_state.analysis_results):
            if "error" in result:
                continue
                
            match_percentage = int(result.get("OverallMatch", "0%").strip("%"))
            
            candidates_list.append({
                "index": i,
                "name": result.get("CandidateName", f"Candidate {i+1}"),
                "match_percentage": match_percentage,
                "recommendation": result.get("Recommendation", "further review"),
                "strengths": result.get("Strengths", []),
                "missing_skills": result.get("MissingSkills", []),
                "contact": result.get("ContactInfo", "Not available")
            })
        
        # Sort based on selected option
        if sort_option == "Match % (High to Low)":
            candidates_list.sort(key=lambda x: x["match_percentage"], reverse=True)
        elif sort_option == "Match % (Low to High)":
            candidates_list.sort(key=lambda x: x["match_percentage"])
        else:  # Name (A-Z)
            candidates_list.sort(key=lambda x: x["name"])
        
        # Display candidates
        for candidate in candidates_list:
            match_class = "match-high" if candidate["match_percentage"] >= 70 else ("match-medium" if candidate["match_percentage"] >= 50 else "match-low")
            
            # Check if candidate is in shortlist
            shortlisted = any(s["name"] == candidate["name"] for s in st.session_state.shortlisted)
            
            # Create candidate card with appropriate styling
            card_class = "candidate-card"
            if shortlisted:
                card_class += " shortlisted"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3 style="margin: 0;">{candidate["name"]}</h3>
                    <span style="font-weight: bold; color: var(--{'success' if match_class == 'match-high' else 'warning' if match_class == 'match-medium' else 'danger'}-color);">
                        {candidate["match_percentage"]}% Match
                    </span>
                </div>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
                    <div><strong>Contact:</strong> {candidate["contact"]}</div>
                    <div><strong>Recommendation:</strong> {candidate["recommendation"].capitalize()}</div>
                </div>
                <div class="match-progress">
                    <div class="match-progress-bar {match_class}" style="width: {candidate["match_percentage"]}%"></div>
                </div>
                
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            # Strengths preview
            strengths_preview = ", ".join(candidate["strengths"][:2])
            if strengths_preview:
                st.markdown(f"**Key Strengths:** {strengths_preview}")
            
            # Action buttons in columns
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                if st.button(f"{'✓ Shortlisted' if shortlisted else 'Shortlist'}", key=f"shortlist_{candidate['index']}", 
                             type="primary" if not shortlisted else "secondary"):
                    # Toggle shortlisting
                    if shortlisted:
                        # Remove from shortlist
                        st.session_state.shortlisted = [s for s in st.session_state.shortlisted if s["name"] != candidate["name"]]
                    else:
                        # Add to shortlist
                        new_shortlist = {
                            "name": candidate["name"],
                            "contact": candidate["contact"],
                            "match_percentage": candidate["match_percentage"],
                            "strengths": candidate["strengths"],
                            "missing_skills": candidate["missing_skills"],
                            "recommendation": candidate["recommendation"]
                        }
                        st.session_state.shortlisted.append(new_shortlist)
                    
                    st.experimental_rerun()
            
            with col2:
                if st.button("View Details", key=f"view_{candidate['index']}"):
                    # Set index for detailed view
                    st.session_state.current_candidate_index = candidate["index"]
                    st.session_state.show_candidate_details = True
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Show shortlisted candidates
        if st.session_state.shortlisted:
            st.markdown("### Shortlisted Candidates")
            
            shortlist_html = ""
            for candidate in st.session_state.shortlisted:
                shortlist_html += f"""
                <div style="background-color: rgba(99, 212, 113, 0.1); border-left: 4px solid var(--success-color); 
                            padding: 0.75rem; margin-bottom: 0.5rem; border-radius: var(--border-radius);">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{candidate["name"]}</strong>
                        <span>{candidate["match_percentage"]}% Match</span>
                    </div>
                </div>
                """
            
            st.markdown(shortlist_html, unsafe_allow_html=True)
            
            # Next step button
            if st.button("Schedule Interviews ➡️", type="primary"):
                next_step()
                st.experimental_rerun()
        else:
            st.info("No candidates have been shortlisted yet. Use the 'Shortlist' button to add candidates.")
        
        # Navigation button
        if st.button("⬅️ Back to Analysis"):
            prev_step()
            st.experimental_rerun()
    else:
        st.warning("No candidate analysis results available. Please go back and process resumes first.")
        
        if st.button("⬅️ Back to Upload Resumes"):
            go_to_step(2)
            st.experimental_rerun()

# Step 5: Schedule Interviews
elif st.session_state.current_step == 5:
    st.markdown("## Step 5: Schedule Interviews")
    
    if not st.session_state.shortlisted:
        st.warning("No candidates have been shortlisted. Please go back and shortlist candidates first.")
        
        if st.button("⬅️ Back to Shortlisting"):
            prev_step()
            st.experimental_rerun()
    else:
        st.markdown("### Schedule Interview Invitations")
        
        # Select candidate to schedule
        candidate_names = [candidate["name"] for candidate in st.session_state.shortlisted]
        selected_candidate_name = st.selectbox("Select Candidate", candidate_names)
        
        # Find the selected candidate
        selected_candidate = next((c for c in st.session_state.shortlisted if c["name"] == selected_candidate_name), None)
        
        if selected_candidate:
            st.session_state.selected_candidate = selected_candidate
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {selected_candidate['name']}")
                st.markdown(f"**Contact:** {selected_candidate['contact']}")
                st.markdown(f"**Match:** {selected_candidate['match_percentage']}%")
                
                strengths_text = ", ".join(selected_candidate['strengths'][:3])
                if strengths_text:
                    st.markdown(f"**Key Strengths:** {strengths_text}")
            
            with col2:
                if st.button("Generate Interview Invitation", type="primary"):
                    with st.spinner("Generating interview invitation..."):
                        # Generate email for candidate
                        email_data = generate_interview_email(selected_candidate, st.session_state.jd_summary)
                        
                        if "error" not in email_data:
                            st.session_state.interview_email = email_data
                            st.success("Interview invitation generated!")
                        else:
                            st.error(f"Error generating invitation: {email_data['error']}")
            
            # Display generated email
            if st.session_state.interview_email:
                st.markdown("### Interview Invitation Preview")
                
                st.markdown('<div class="email-preview">', unsafe_allow_html=True)
                st.markdown(f"**To:** {st.session_state.interview_email['candidate_email']}")
                st.markdown(f"**Subject:** {st.session_state.interview_email['email_subject']}")
                st.markdown("**Body:**")
                st.markdown(st.session_state.interview_email['email_body'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Email sending options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Copy to Clipboard"):
                        # Generate text for clipboard
                        email_text = f"""To: {st.session_state.interview_email['candidate_email']}
Subject: {st.session_state.interview_email['email_subject']}

{st.session_state.interview_email['email_body']}"""
                        
                        # Use streamlit to copy to clipboard (this is a placeholder - actual clipboard functionality requires JavaScript)
                        st.code(email_text)
                        st.success("Email copied to clipboard! Use Ctrl+C to copy the text above.")
                
                with col2:
                    # Generate mailto link
                    mailto_data = generate_mailto_link(st.session_state.interview_email)
                    
                    if mailto_data["status"] == "success":
                        mailto_link = mailto_data["mailto_link"]
                        st.markdown(f'<a href="{mailto_link}" target="_blank"><button style="background-color: var(--success-color); color: white; padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; width: 100%;">Open in Email Client</button></a>', unsafe_allow_html=True)
                    else:
                        st.error(mailto_data["message"])
        
        # Navigation
        if st.button("⬅️ Back to Shortlisting"):
            prev_step()
            st.experimental_rerun()
        
        # Show candidate scheduling progress
        st.markdown("### Interview Scheduling Progress")
        
        # Create a progress table
        progress_data = []
        for candidate in st.session_state.shortlisted:
            # Check if an email has been generated for this candidate
            email_sent = (st.session_state.selected_candidate and 
                         st.session_state.selected_candidate["name"] == candidate["name"] and 
                         st.session_state.interview_email is not None)
            
            progress_data.append({
                "name": candidate["name"],
                "email": candidate["contact"],
                "match": f"{candidate['match_percentage']}%",
                "status": "Email Generated" if email_sent else "Pending"
            })
        
        # Convert to DataFrame for display
        if progress_data:
            import pandas as pd
            df = pd.DataFrame(progress_data)
            st.dataframe(df, use_container_width=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee; color: #777;">
    EzHunt AI Recruitment Assistant | Powered by Generative AI | © 2025
</div>
""", unsafe_allow_html=True)