from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Aharix AI personality
SYSTEM_PROMPT = """
You are Ahar AI, the nutrition assistant of the Aharix application.

Your purpose is to help users understand:
- food ingredients
- nutrition labels
- harmful additives
- allergens
- sugars and fats
- healthier alternatives

You ONLY answer food and nutrition related questions.

If a user asks something unrelated (coding, programming, math, etc),
politely say:

"I'm Ahar AI, a nutrition assistant inside the Aharix app.
I can only help with food ingredients, nutrition, and healthier eating choices."
"""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {"status": "Ahar AI backend running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    user_message = request.message

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    reply = response.output_text or "No response from AI."

    return ChatResponse(reply=reply)