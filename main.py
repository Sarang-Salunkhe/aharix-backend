from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Ahar AI, the nutrition assistant of the Aharix app.

Your job is to help users understand food products and ingredients.

You can explain:
• ingredients in food
• nutrition labels
• sugars, fats, additives
• allergens
• health risks
• healthier alternatives

If a user asks something unrelated to food or nutrition (coding, math, general questions),
politely respond:

"I'm Ahar AI from the Aharix app. I can only help with food ingredients, nutrition, and healthier eating choices."
"""

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "Aharix AI backend running"}


@app.post("/chat")
def chat(request: ChatRequest):

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ]
    )

    return {"reply": response.output_text}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What food product is this? Is it healthy?"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    return {"reply": response.output_text}
    
    
    
