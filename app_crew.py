import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from PIL import Image
import openai
from crewai import Agent, Task, Crew

# Load data
df = pd.read_csv("data.csv")

# Configure page
st.set_page_config(page_title="AI Social Welfare Assistant", layout="wide")
st.title("🌟 AI-Powered Social Welfare Assistant")

# File processing with strict data extraction
def process_application(pdf_file, img_file, audio_file):
    """Process files with validation"""
    results = {
        'pdf_text': '',
        'id_verified': False,
        'transcript': '',
        'sentiment': 'Neutral'
    }
    
    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
            results['pdf_text'] = " ".join([page.extract_text() for page in reader.pages])[:500] + "..."
        except Exception as e:
            st.error(f"PDF Error: {str(e)}")

    if img_file:
        try:
            Image.open(img_file).verify()
            results['id_verified'] = True
        except Exception as e:
            results['id_verified'] = False
            st.error(f"Image Error: {str(e)}")

    if audio_file:
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            audio_file.seek(0)
            transcript = client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read(), "audio/wav"),
                model="whisper-1",
                response_format="text"
            )
            results['transcript'] = transcript[:500] + "..."
            results['sentiment'] = 'Positive' if 'help' in transcript.lower() else 'Neutral'
        except Exception as e:
            st.error(f"Audio Error: {str(e)}")

    return results

# Constrained Decision Agent
decision_agent = Agent(
    role="Strict Eligibility Officer",
    goal="Make decisions based ONLY on provided data",
    backstory="Expert in welfare policy adherence",
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    temperature=0.1  # Reduce creativity
)

def create_decision_task(applicant_data, processed_data):
    """Create task with strict constraints"""
    return Task(
        description=f"""
        Analyze ONLY this data:
        - Name: {applicant_data['name']}
        - Income: ${applicant_data['income']}
        - Dependents: {applicant_data['dependents']}
        - ID Verified: {processed_data['id_verified']}
        - Document Content: {processed_data['pdf_text'][:200]}
        - Voice Sentiment: {processed_data['sentiment']}
        
        Apply EXACTLY these rules:
        1. ID must be verified
        2. Income must be under $400
        3. At least 2 dependents
        """,
        agent=decision_agent,
        expected_output="""
        Structured decision in this format:
        ## Eligibility Result
        **Status:** Approved/Rejected  
        **Reason:** [concise reason based on rules]
        
        ## Verification Summary
        - ID Verified: Yes/No
        - Income: $[amount] [meets/doesn't meet] requirement
        - Dependents: [number] [meets/doesn't meet] requirement
        """,
        context=[],  # Prevent external knowledge
        config={"temperature": 0.0}  # Maximize determinism
    )

# Streamlit UI
with st.sidebar:
    st.header("📥 Application Inputs")
    openai_key = st.text_input("🔑 OpenAI API Key", type="password")
    selected_applicant = st.selectbox("👤 Select Applicant", df['name'].unique())
    pdf_file = st.file_uploader("📑 Supporting Document (PDF)", type=["pdf"])
    img_file = st.file_uploader("🖼️ ID Verification", type=["jpg", "jpeg", "png"])
    audio_file = st.file_uploader("🔊 Voice Explanation", type=["mp3", "wav"])
    analyze_btn = st.button("🚀 Analyze Application")

if analyze_btn:
    if not openai_key:
        st.error("🔑 API key required")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = openai_key
    applicant = df[df['name'] == selected_applicant].iloc[0]
    
    processed_data = process_application(pdf_file, img_file, audio_file)
    
    try:
        decision_task = create_decision_task(applicant, processed_data)
        crew = Crew(agents=[decision_agent], tasks=[decision_task])
        decision_result = crew.kickoff()
    except Exception as e:
        st.error(f"Decision Error: {str(e)}")
        st.stop()

    # Display Results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Applicant Profile")
        st.markdown(f"""
        **Name:** {applicant['name']}  
        **Income:** ${applicant['income']}  
        **Dependents:** {applicant['dependents']}
        """)
        
        if img_file:
            st.image(Image.open(img_file), width=250)
            status = "✅ Verified" if processed_data['id_verified'] else "❌ Unverified"
            st.markdown(f"**ID Status:** {status}")

    with col2:
        if pdf_file:
            st.subheader("📄 Document Excerpt")
            st.write(processed_data['pdf_text'])
        
        if audio_file:
            st.subheader("🗣️ Voice Analysis")
            st.write(processed_data['transcript'])
            st.markdown(f"**Sentiment:** {processed_data['sentiment']}")

    st.markdown("---")
    st.subheader("📝 Final Decision")
    st.markdown(decision_result)

else:
    st.info("💡 Select an applicant and upload documents to begin analysis")