import streamlit as st
import os
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from utils import apply_css, initialize_session, get_model_name, create_chat_session, analyze_response, detect_sentiment
from config import Config

def main():
    st.set_page_config(page_title="ChatDisc with Gemini Models", layout="wide")

    initialize_session()

    st.sidebar.title("Settings")
    st.session_state['dark_mode'] = st.sidebar.checkbox("Dark Mode", value=st.session_state['dark_mode'])

    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Gemini 1.5 Flash", "Gemini 1.5 Pro", "Gemini 1.0 Pro"]
    )

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 1.0)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95)
    top_k = st.sidebar.slider("Top K", 0, 100, 64)
    max_output_tokens = st.sidebar.slider("Max Output Tokens", 1, 8192, 8192)

    apply_css(st.session_state['dark_mode'])

    st.title("Chatbot with Gemini Models")

    api_key = st.text_input("Enter your Gemini API key:", type="password")
    if api_key:
        model_name = get_model_name(model_choice)
        chat_session = create_chat_session(api_key, model_name, temperature, top_p, top_k, max_output_tokens)

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        use_voice = st.checkbox("Use Voice Input")

        if use_voice:
            if st.button("Start Listening"):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("Listening...")
                    audio = recognizer.listen(source)
                    try:
                        user_input = recognizer.recognize_google(audio)
                        st.write(f"You said: {user_input}")

                        # Send the recognized user input to the chatbot
                        if user_input:
                            response = chat_session.send_message(user_input)
                            st.session_state.chat_history.append(("user", user_input))
                            st.session_state.chat_history.append(("ai", response.text, model_name))

                            # Convert response to speech and display media player
                            tts = gTTS(text=response.text, lang='en')
                            audio_file = "response.mp3"
                            tts.save(audio_file)
                            st.audio(audio_file, format='audio/mp3')

                    except sr.UnknownValueError:
                        st.write("Sorry, I did not understand that.")
                    except sr.RequestError as e:
                        st.write(f"Could not request results; {e}")

        user_input = st.text_input("You: ", key="user_input")
        if user_input:
            response = chat_session.send_message(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("ai", response.text, model_name))

            # Convert response to speech and display media player
            tts = gTTS(text=response.text, lang='en')
            audio_file = "response.mp3"
            tts.save(audio_file)
            st.audio(audio_file, format='audio/mp3')

        chat_container = st.container()
        with chat_container:
            for entry in st.session_state.chat_history:
                if entry[0] == "user":
                    st.markdown(f"""
                        <div class="chat-message user">
                            <div class="chat-icon user"></div>
                            <div class="chat-bubble user">{entry[1]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    sentiment = detect_sentiment(entry[1])
                    feedback = analyze_response(entry[1])
                    st.markdown(f"""
                        <div class="chat-message ai">
                            <div class="chat-icon ai"></div>
                            <div class="chat-bubble ai">
                                {entry[1]} <br>
                                <small>{entry[2]}</small> <br>
                                <small>Sentiment: {sentiment}</small> <br>
                                <small>Feedback: {feedback}</small>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
