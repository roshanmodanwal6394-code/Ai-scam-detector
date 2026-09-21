"""
llm_analyzer.py
LangChain integration module enforcing structured schema extraction
via Google Gemini API.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class ScamIndicators(BaseModel):
    urgency_score: float = Field(..., ge=0.0, le=10.0)
    suspicious_link_score: float = Field(..., ge=0.0, le=10.0)
    sensitive_info_score: float = Field(..., ge=0.0, le=10.0)
    threat_reward_score: float = Field(..., ge=0.0, le=10.0)
    impersonation_target: Optional[str] = None
    key_red_flags: List[str]
    linguistic_reasoning: str
    safety_recommendations: List[str]

def analyze_message_with_llm(message_text: str, api_key: Optional[str] = None) -> ScamIndicators:
    active_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not active_key:
        raise ValueError("Google API Key not detected.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=active_key,
    )

    structured_llm = llm.with_structured_output(ScamIndicators)

    system_prompt = (
        "You are a cyber security intelligence specialist. Analyze the incoming communication "
        "for social engineering attacks, digital extortion, phishing, and lottery scams.\n"
        "Calibrate each numeric score precisely between 0.0 (completely benign) and 10.0 (extreme threat)."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze this message carefully:\n\n\"{message_to_analyze}\"")
    ])

    extraction_chain = prompt_template | structured_llm
    return extraction_chain.invoke({"message_to_analyze": message_text})
