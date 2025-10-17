from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables 
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- LangChain Setup ---
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",  # Fast, efficient model
    "llama-3.3-70b-versatile",  # High-quality translations
    "mixtral-8x7b-32768",    # Larger context model
]

DEFAULT_MODEL = SUPPORTED_MODELS[0]
chain = None

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    selected_model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    if selected_model not in SUPPORTED_MODELS:
        logger.warning(f"Model {selected_model} not in supported list, falling back to {DEFAULT_MODEL}")
        selected_model = DEFAULT_MODEL

    logger.info(f"Initializing Groq with model: {selected_model}")
    model = ChatGroq(
        model=selected_model,
        groq_api_key=groq_api_key,
        temperature=0.1  # Lower temperature for more consistent translations
    )

    # Improved system prompt for better translations
    system_template = """You are an expert translator. 
    Translate the following text into {language}. 
    Provide only the translated text without explanations or notes."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}")
    ])

    parser = StrOutputParser()
    chain = prompt_template | model | parser
    logger.info(f"LangChain initialized successfully with {selected_model}")

except Exception as e:
    logger.exception("Failed to initialize LangChain")
    chain = None

# --- FastAPI App ---
app = FastAPI(
    title="TransNova Translation Service",
    version="1.1",
    description="Neural machine translation API powered by Groq LLMs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    language: str
    text: str

@app.get("/health")
async def health():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy" if chain else "degraded",
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        "chain_initialized": chain is not None,
        "api_key_configured": bool(os.getenv("GROQ_API_KEY"))
    }

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Handle translation requests with improved error handling"""
    if not chain:
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - model not initialized"
        )

    try:
        logger.info(f"Processing translation to {request.language}: {request.text[:50]}...")
        
        # Execute translation in thread pool
        result = await asyncio.to_thread(
            chain.invoke,
            {"language": request.language, "text": request.text}
        )

        if not result or not isinstance(result, str) or not result.strip():
            raise ValueError("Empty or invalid translation result")

        logger.info("Translation successful")
        return {"translation": result.strip()}

    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")