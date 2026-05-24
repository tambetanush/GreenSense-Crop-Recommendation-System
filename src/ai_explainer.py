import os
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class AIExplanation(BaseModel):
    summary: str
    key_factors: list[str]
    cautions: list[str]

def get_ai_explanation(sensor_readings: dict, explainability_output: dict, crop_name: str) -> AIExplanation:
    """
    Calls the Gemini API to explain the recommendation, utilizing up to 3 API keys
    for rotation and handling rate limits (429) or server errors (5xx).
    Returns an AIExplanation Pydantic model.
    """
    
    # 1. Fetch available keys
    keys = []
    for key_name in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        key_val = os.environ.get(key_name)
        if key_val:
            keys.append(key_val)
            
    if not keys:
        # Fallback to the standard GEMINI_API_KEY if none of the numbered ones exist
        fallback = os.environ.get("GEMINI_API_KEY")
        if fallback:
            keys.append(fallback)
        else:
            raise ValueError("No Gemini API keys found in environment. Please set GEMINI_API_KEY_1, etc.")

    # Determine overall score to adapt prompt
    overall_score = None
    if "ml_score" in explainability_output and explainability_output["ml_score"] is not None:
        overall_score = explainability_output["ml_score"]
    elif "cosine_score" in explainability_output and explainability_output["cosine_score"] is not None:
        overall_score = explainability_output["cosine_score"]

    if overall_score is not None and overall_score < 0.6:
        guidance = "The recommendation score for this crop is relatively LOW. Focus on actionable advice: explain exactly WHAT MUST BE DONE (e.g., adding specific fertilizers, adjusting pH, waiting for a different season) to improve the suitability of the environment for this crop."
    else:
        guidance = "The recommendation score for this crop is HIGH. Focus on scientific reasoning: explain the biological and environmental reasons why this crop thrives in these exact conditions."

    system_prompt = f"""You are an expert agricultural AI assistant. 
Your task is to explain why the crop '{crop_name}' was recommended (or evaluated) given the current environmental conditions.

{guidance}

Always structure your response strictly according to the provided JSON schema:
- summary: A short, concise overview of the crop's compatibility.
- key_factors: A list of 2-3 bullet points highlighting the most important positive or negative factors based on the sensor readings.
- cautions: A list of 1-2 bullet points highlighting potential risks or areas requiring attention.

Keep the language professional, accessible to farmers, and scientifically accurate.
"""

    prompt_content = f"""
Crop Name: {crop_name}

Sensor Readings:
{sensor_readings}

System's Technical Explainability Data:
{explainability_output}
"""

    # 2. Try the request with up to 3 retries, rotating keys
    max_retries = min(3, len(keys))
    last_exception = None

    for attempt in range(max_retries):
        current_key = keys[attempt % len(keys)]
        
        try:
            client = genai.Client(api_key=current_key)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=AIExplanation,
                    temperature=0.3
                )
            )
            
            # The response.text is guaranteed to be a JSON string matching the schema
            return AIExplanation.model_validate_json(response.text)
            
        except APIError as e:
            # Catch API errors (like 429 rate limit or 5xx server errors)
            print(f"[AI Explainer] Attempt {attempt+1} failed with key {attempt+1}. Error: {e}")
            last_exception = e
            continue
        except Exception as e:
            # For validation errors or other non-API errors, we might not want to retry,
            # but for robustness we'll log it and continue.
            print(f"[AI Explainer] Attempt {attempt+1} failed with unknown error: {e}")
            last_exception = e
            continue
            
    # If we exhaust all retries
    raise Exception(f"Failed to generate AI explanation after {max_retries} attempts. Last error: {last_exception}")
