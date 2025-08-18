from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- LangChain Setup ---
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
        
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    system_template = "Translate the following into {language}:"
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        ('user', '{text}')
    ])

    parser = StrOutputParser()
    chain = prompt_template | model | parser
except Exception as e:
    print(f"LangChain setup error: {e}")
    chain = None

# --- FastAPI App ---
app = FastAPI(
    title="Langchain Translation Server",
    version="1.0",
    description="A simple API server for text translation."
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    language: str
    text: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    if not chain:
        return {"error": "LangChain model not initialized."}
    try:
        result = chain.invoke({"language": request.language, "text": request.text})
        return {"translation": result}
    except Exception as e:
        return {"error": str(e)}