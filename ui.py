import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Text Translator",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- UI Elements ---
st.title("📝 Text Translator UI")
st.markdown("This UI connects to a FastAPI backend to translate text using a Large Language Model.")

# Input text from user
st.header("Enter the text you want to translate")
text_to_translate = st.text_area("Text", "Hello, how are you today?", height=150)

# Dropdown for language selection
st.header("Select the target language")
language_options = ["French", "Spanish", "German", "Japanese", "Hindi", "Italian"]
selected_language = st.selectbox("Language", language_options)

# Define the API endpoint (Keeping the user's hardcoded URL)
api_url = "https://transnova-1.onrender.com/translate"

# Button to trigger translation
if st.button("Translate"):
    if text_to_translate and selected_language:
        
        # Create the JSON payload to send to the API
        payload = {
            "language": selected_language,
            "text": text_to_translate
        }
        
        try:
            # Show a spinner while waiting for the response
            with st.spinner(f"Translating to {selected_language}..."):
                # Send POST request to the FastAPI backend
                response = requests.post(api_url, json=payload)
                
                # Check if the request was successful (Status code 200)
                if response.status_code == 200:
                    translation_result = response.json()
                    
                    if "translation" in translation_result:
                        # Case 1: Success - Key is present
                        translated_text = translation_result["translation"]
                        st.success("Translation successful!")
                        st.subheader("Translated Text:")
                        st.text_area("Result", value=translated_text, height=150, disabled=True)
                    else:
                        # Case 2: Success status but malformed body (Your current issue)
                        st.error("Translation failed: Backend returned a successful status (200) but the expected 'translation' key was missing.")
                        st.text_area("Raw API Response for Debugging", value=str(translation_result), height=100, disabled=True)
                        st.info("Check your FastAPI logs for possible errors related to the LangChain model or API key.")
                        
                else:
                    # Case 3: API Error (4xx, 5xx)
                    st.error(f"Error from API: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            # Case 4: Connection Error
            st.error(f"Could not connect to the backend API at {api_url}.")
            st.error(f"Please check if the FastAPI server is running and accessible.")
            st.info(f"Connection Details: {e}")
            
    else:
        st.warning("Please enter text to translate and select a language.")