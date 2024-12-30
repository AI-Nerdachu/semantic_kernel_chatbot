import streamlit as st
import requests

st.title("Seamless Chatbot Demonstration")

# Chat endpoint
CHAT_ENDPOINT = "http://127.0.0.1:8000/chat"

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.text_input("You:", key="user_input")
if st.button("Send") and user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Send message to chatbot
    try:
        response = requests.post(CHAT_ENDPOINT, json={"message": user_input})
        if response.status_code == 200:
            bot_response = response.json()["response"]
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})
        else:
            st.session_state["messages"].append({"role": "assistant", "content": "Error: Chatbot is unavailable."})
    except requests.exceptions.RequestException as e:
        st.session_state["messages"].append({"role": "assistant", "content": f"Error: {str(e)}"})

# Display conversation
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Chatbot:** {message['content']}")
