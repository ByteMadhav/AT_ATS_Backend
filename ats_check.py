import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
os.environ["DEBUG"] = os.getenv("DEBUG", "")

# Using the standard OpenAI SDK to call Gemini
model = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """You are an expert ATS scanner.
Evaluate the resume and return a structured JSON response with keys:
- "score": Integer (0-100)
- "summary": Short string
- "strengths": List of strings
- "weaknesses": List of strings
- "suggestions": List of strings."""


async def evaluate_resume(resume_text: str, job_description: str = "") -> dict:
    prompt = f"Resume:\n{resume_text}\n"
    if job_description:
        prompt += f"\nJob Description:\n{job_description}\n"

    response = await model.chat.completions.create(
        model="gemini-3.6-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        max_tokens=1500,
        timeout=30.0,
        top_p=1.0,
        temperature=0.0
        # REMOVED: seed=42 (Not supported by Gemini's OpenAI wrapper)
        # REMOVED: top_k=1 (Not standard for OpenAI SDK, causes similar errors)
    )

    return json.loads(response.choices[0].message.content)