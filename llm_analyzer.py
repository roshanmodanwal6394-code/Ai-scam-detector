"""
llm_analyzer.py
LangChain integration module for Google Gemini API.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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


def analyze_message_with_llm(
    message_text: str,
    api_key: Optional[str] = None
) -> ScamIndicators:

    active_key = api_key or os.getenv("GOOGLE_API_KEY")

    if not active_key:
        raise ValueError("Google API Key not detected.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=active_key,
    )

    structured_llm = llm.with_structured_output(ScamIndicators)

    system_prompt = """
You are a cybersecurity intelligence specialist.

Analyze the provided communication for:
- phishing
- social engineering
- digital extortion
- lottery scams
- impersonation
- suspicious links
- requests for sensitive information

Give numeric scores from 0.0 to 10.0:
0.0 = completely benign
10.0 = extreme threat

Return only information matching the requested structured schema.
"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                'Analyze this message carefully:\n\n"{message_to_analyze}"'
            ),
        ]
    )

    extraction_chain = prompt_template | structured_llm

    result = extraction_chain.invoke(
        {"message_to_analyze": message_text}
    )

    return result
