import base64

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        

def generate_response():
    # Display bot message in chat message container
    with st.chat_message("assistant"):
        placeholder = st.empty()

    # Generate bot response
    full_response = { "type": "text", "text": "" }
    messages = st.session_state.messages

    if st.session_state.system_message != "":
        messages = [SystemMessage(content=st.session_state.system_message)] + messages

    message = AIMessage(content=[full_response])
    st.session_state.messages.append(message)

    try:
        for response in st.session_state.chatbot.stream(messages):
            full_response["text"] += response.content
            message.content = [full_response]
            placeholder.markdown(full_response["text"] + "▌")
        placeholder.markdown(full_response["text"])
    except openai.AuthenticationError as e:
        st.error("Invalid OpenAI API key.")


def get_role(lc_message):
    return {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system"
    }[lc_message.__class__]


def print_message(lc_message):
    role = get_role(lc_message)
    if role != "system":
        with st.chat_message(role):
            if lc_message.content[0]["type"] == "text":
                st.markdown(lc_message.content[0]['text'])
            else:
                st.image(lc_message.content[0]["image_url"]["url"])


def st_chatbot(openai_api_key):
    if openai_api_key == st.secrets["PASSWORD"]:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    elif openai_api_key == "":
        st.warning("Please enter your OpenAI API key.")
        st.stop()
        
    if ("chatbot" not in st.session_state) or (openai_api_key != st.session_state.chatbot.openai_api_key):
        st.session_state.chatbot = ChatOpenAI(max_tokens=2048, openai_api_key=openai_api_key)
                
    openai_models = st.sidebar.radio("Select OpenAI model:", ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-vision-preview"], index=1)
    st.session_state.chatbot.model_name = openai_models

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    with st.sidebar:
        st.text_area("System message", key="system_message", value="")
        st.session_state.chatbot.temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        st.button("Reset chat", on_click=lambda: st.session_state.pop("messages", None), type="primary", use_container_width=True)
        regenerate = st.button("Regenerate response", on_click=lambda: st.session_state.messages.pop(), use_container_width=True)
        st.button("Delete last", on_click=lambda: st.session_state.messages.pop(), use_container_width=True)
        stop = st.button("Stop", use_container_width=True)

        if "vision" in openai_models:
            with st.form("my-form", clear_on_submit=True):
                uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="image")
                submitted = st.form_submit_button("UPLOAD!")

            if submitted:
                base64_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
                content = [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }]
                
                message = HumanMessage(content=content)
                st.session_state.messages.append(message)


    st.title("ChatGPT UI")


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        print_message(message)
        

    # React to user input
    if (prompt := st.chat_input("What is up?")) and not stop:
        message = HumanMessage(content=[{
                "type": "text",
                "text": prompt,
        }])
        st.session_state.messages.append(message)
        print_message(message)
        generate_response()
    elif regenerate and not stop:
        generate_response()