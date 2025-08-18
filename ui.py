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

# Button to trigger translation
if st.button("Translate"):
    if text_to_translate and selected_language:
        # Define the API endpoint
        api_url = "https://transnova-1.onrender.com/translate"
        
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
                
                # Check if the request was successful
                if response.status_code == 200:
                    translation_result = response.json()
                    st.success("Translation successful!")
                    st.subheader("Translated Text:")
                    # Use a text area for better formatting of long text
                    st.text_area("Result", value=translation_result.get("translation", "No translation found."), height=150, disabled=True)
                else:
                    # Handle API errors
                    st.error(f"Error from API: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            # Handle connection errors
            st.error(f"Could not connect to the backend API. Please ensure the FastAPI server is running.")
            st.error(f"Details: {e}")
            
    else:
        st.warning("Please enter text to translate and select a language.")

