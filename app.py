from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. LangChain Setup ---
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
        
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    # Create prompt template
    system_template = "Translate the following into {language}:"
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        ('user', '{text}')
    ])

    # Create a simple parser
    parser = StrOutputParser()

    # Create the final chain
    chain = prompt_template | model | parser

except Exception as e:
    print(f"An error occurred during LangChain setup: {e}")
    # Exit or handle the error as appropriate
    chain = None 

# --- 2. FastAPI App Definition ---
app = FastAPI(
    title="Langchain Translation Server",
    version="1.0",
    description="A simple API server for text translation."
)

# Pydantic model for input validation
class TranslationRequest(BaseModel):
    language: str
    text: str

# --- 3. API Endpoint ---
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Receives text and a target language, returns the translation.
    """
    if not chain:
        return {"error": "LangChain model not initialized. Check server logs."}
        
    try:
        # Manually invoke the chain with the request data
        result = chain.invoke({"language": request.language, "text": request.text})
        return {"translation": result}
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return {"error": "Failed to process the translation request."}

# This part is for running with uvicorn from the command line
# It's good practice to keep it for reference
if __name__ == "__main__":
    import uvicorn
    # Note: It's better to run this from the terminal with `uvicorn app:app --reload`
    uvicorn.run(app, host="127.0.0.1", port=8000)
