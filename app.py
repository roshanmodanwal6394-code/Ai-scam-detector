"""
app.py
Streamlit Application orchestrating LangChain inference and Fuzzy System.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from llm_analyzer import analyze_message_with_llm
from fuzzy_logic import calculate_scam_risk

load_dotenv()

st.set_page_config(page_title="AI Scam Message Detector", page_icon="🛡️", layout="wide")

st.sidebar.title("⚙️ Engine Configuration")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    value=os.getenv("GOOGLE_API_KEY", ""),
    help="Acquire an API Key from Google AI Studio."
)

sample_choice = st.sidebar.selectbox(
    "Select benchmark sample:",
    [
        "Custom Input",
        "Sample 1: Fake Bank Urgent KYC (High Risk)",
        "Sample 2: Jio KBC Lottery Winning (High Risk)",
        "Sample 3: Courier Delivery Rescheduling (Suspicious)",
        "Sample 4: Casual Student Meeting (Safe)"
    ]
)

sample_data = {
    "Sample 1: Fake Bank Urgent KYC (High Risk)": (
        "URGENT: Your SBI Bank account #4829 has been locked due to unverified KYC. "
        "Visit http://sbi-kyc-verify-portal.in immediately to update Aadhaar and submit your 6-digit OTP. "
        "Account will be permanently deactivated in 30 minutes."
    ),
    "Sample 2: Jio KBC Lottery Winning (High Risk)": (
        "Congratulations! Your phone number won ₹25,00,000 in the KBC WhatsApp Lucky Draw. "
        "Send your bank passbook image and processing fee of ₹2,500 to WhatsApp number 9876543210 to claim today."
    ),
    "Sample 3: Courier Delivery Rescheduling (Suspicious)": (
        "Notice: BlueDart package tracking number BL9022 is stuck at the distribution warehouse. "
        "Update your delivery address pin code at http://tinyurl.com/bd-pkg-update to resume transit."
    ),
    "Sample 4: Casual Student Meeting (Safe)": (
        "Hey Priya, did you complete the Fuzzy Logic assignment for the Internal Assessment? "
        "Let's meet at the library by 3 PM to review our presentations."
    )
}

default_text = sample_data.get(sample_choice, "")

st.title("🛡️ AI Scam Message Detection System")
st.markdown("**Hybrid Architecture:** LangChain & Gemini + Mamdani Fuzzy Inference")
st.markdown("---")

message_input = st.text_area(
    "📥 Enter Suspicious Message for Forensic Assessment:",
    value=default_text,
    height=130,
    placeholder="Paste message text..."
)

if st.button("🚀 Analyze Message Risk", type="primary", use_container_width=True):
    if not message_input.strip():
        st.warning("⚠️ Please provide text inside the input box to evaluate.")
    elif not api_key_input.strip():
        st.error("🔑 Google Gemini API key required.")
    else:
        with st.spinner("Analyzing message and running fuzzy inference..."):
            try:
                indicators = analyze_message_with_llm(message_input, api_key=api_key_input)
                fuzzy_score = calculate_scam_risk(
                    indicators.urgency_score,
                    indicators.suspicious_link_score,
                    indicators.sensitive_info_score,
                    indicators.threat_reward_score
                )

                if fuzzy_score >= 65.0:
                    badge_style = "🚨 HIGH RISK"
                elif fuzzy_score >= 35.0:
                    badge_style = "⚠️ SUSPICIOUS"
                else:
                    badge_style = "✅ SAFE"

                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.subheader("📊 Fuzzy Logic Evaluation")
                    st.metric(label="Calculated Fuzzy Risk Score", value=f"{fuzzy_score} / 100", delta=badge_style)
                    st.progress(indicators.urgency_score / 10.0, text=f"Urgency: {indicators.urgency_score}/10")
                    st.progress(indicators.suspicious_link_score / 10.0, text=f"Link Suspicion: {indicators.suspicious_link_score}/10")
                    st.progress(indicators.sensitive_info_score / 10.0, text=f"Sensitive Info: {indicators.sensitive_info_score}/10")
                    st.progress(indicators.threat_reward_score / 10.0, text=f"Threat / Reward: {indicators.threat_reward_score}/10")
                    if indicators.impersonation_target:
                        st.info(f"🏛️ **Impersonation Target:** {indicators.impersonation_target}")

                with col_right:
                    st.subheader("🔍 Identified Red Flags")
                    for flag in indicators.key_red_flags:
                        st.markdown(f"- 🚩 **{flag}**")
                    st.markdown("**Reasoning:**")
                    st.write(indicators.linguistic_reasoning)

                st.markdown("---")
                st.subheader("💡 Recommended Actions")
                for rec in indicators.safety_recommendations:
                    st.markdown(f"- 🛡️ {rec}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("⚠️ **Educational Disclaimer:** For academic demonstration only.")
