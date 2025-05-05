import streamlit as st
import pandas as pd
import tempfile
from openai import OpenAI
from PyPDF2 import PdfReader
from PIL import Image
import base64
import os

# Load static CSV data
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Static data file not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Configure page
st.set_page_config(page_title="AI Social Welfare Assistant", layout="wide")
st.title("🌟 AI-Powered Social Welfare Assistant")
st.markdown("""
<style>
    .reportview-container {background: #f5f5f5}
    .sidebar .sidebar-content {background: #ffffff}
    div[data-testid="stHorizontalBlock"] {background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)}
</style>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("📥 Application Inputs")
    st.markdown("---")
    
    # OpenAI API key input
    openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
    
    # Applicant selection dropdown
    selected_applicant = st.selectbox(
        "👤 Select Applicant",
        options=df['name'].unique(),
        index=0
    )
    
    # File uploaders
    pdf_file = st.file_uploader("📑 Supporting Document (PDF)", type=["pdf"])
    img_file = st.file_uploader("🖼️ ID Verification", type=["jpg", "jpeg", "png"])
    audio_file = st.file_uploader("🔊 Voice Explanation", type=["mp3", "wav"])
    
    st.markdown("---")
    analyze_button = st.button("🚀 Analyze Application", use_container_width=True)

# Main content area
if analyze_button:
    # Validate OpenAI API key
    if not openai_key:
        st.error("🔑 OpenAI API key is required to proceed")
        st.stop()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_key)
    
    # Get selected applicant data
    applicant = df[df['name'] == selected_applicant].iloc[0]
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("👤 Applicant Profile")
            st.markdown(f"""
            **Name:** {applicant['name']}  
            **Income:** ${applicant['income']}  
            **Dependents:** {applicant['dependents']}  
            **Status:** {'Approved' if applicant['income'] < 500 else 'Pending'}
            """)
            
            if img_file:
                st.markdown("---")
                st.subheader("🖼️ ID Verification")
                st.image(Image.open(img_file), width=250)
                st.success("✅ ID Validation Successful")

        with col2:
            # Audio processing
            transcript = ""
            if audio_file:
                with st.spinner("🔊 Analyzing voice explanation..."):
                    audio_file.seek(0)
                    transcript = client.audio.transcriptions.create(
                        file=("audio.wav", audio_file.read(), "audio/wav"),
                        model="whisper-1",
                        response_format="text"
                    )
                
                st.subheader("🗣️ Voice Analysis")
                with st.expander("View Full Transcription"):
                    st.write(transcript)
                st.markdown(f"**Key Sentiment:** {'Positive' if 'help' in transcript.lower() else 'Neutral'}")

            # PDF processing
            if pdf_file:
                st.markdown("---")
                st.subheader("📄 Document Analysis")
                reader = PdfReader(pdf_file)
                pdf_text = " ".join([page.extract_text() for page in reader.pages])
                with st.expander("View Document Summary"):
                    st.write(pdf_text[:500] + "...")

    # Decision Logic
    st.markdown("---")
    with st.container():
        col1, col2, col3 = st.columns([1,1,2])
        
        with col1:
            st.subheader("📊 Financial Status")
            st.metric("Monthly Income", f"${applicant['income']}", 
                      delta="Below Threshold" if applicant['income'] < 500 else "Above Threshold")
            
        with col2:
            st.subheader("👨👩👧👦 Dependents")
            st.metric("Family Members", applicant['dependents'], 
                      delta="Meets Requirement" if applicant['dependents'] >= 2 else "Below Requirement")
            
        with col3:
            st.subheader("📝 Final Decision")
            eligible = (applicant['income'] < 500) and (applicant['dependents'] >= 2)
            if eligible:
                st.success("✅ APPROVED FOR SUPPORT")
                st.markdown("**Recommended Support Package:**")
                st.markdown("- Direct cash transfer: $300/month  \n- Job training program  \n- Healthcare benefits")
            else:
                st.error("❌ NOT ELIGIBLE AT THIS TIME")
                st.markdown("**Recommendations:**")
                st.markdown("- Apply for food assistance program  \n- Visit career counseling center")

    # Download Report
    st.markdown("---")
    with st.expander("📥 Download Full Report"):
        summary = f"""
        Applicant Summary Report
        ------------------------
        Name: {applicant['name']}
        Income: ${applicant['income']}
        Dependents: {applicant['dependents']}
        Voice Summary: {transcript[:200] if transcript else 'N/A'}
        Document Notes: {pdf_text[:200] if pdf_file else 'N/A'}
        Final Decision: {'Approved' if eligible else 'Not Approved'}
        """
        
        b64 = base64.b64encode(summary.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{applicant["name"].replace(" ", "_")}_report.txt">Download Text Report</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("💡 Select an applicant and upload supporting documents to begin analysis")
    with st.expander("📋 View Full Applicant Database"):
        st.dataframe(df.style.highlight_max(subset=['income'], color='#fffd75'), use_container_width=True)
