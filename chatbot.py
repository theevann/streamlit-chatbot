import base64

import openai
import streamlit as st
import tiktoken
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


model_costs_multiplier = {
    "gpt-4o-mini": 0,
    "gpt-4o": 0,
    "gpt-4.1": 0,
    "o1-mini": 0,
    "o3-mini": 0,
    "o3": 0,
    "llama3-8b-8192": 0,
    "llama3-70b-8192": 0,
}

class ChatBot():
    def __init__(self, api_key, model_name):
        if api_key == st.secrets["PASSWORD"]:
            if "gpt" in model_name:
                api_key = st.secrets["OPENAI_API_KEY"]
            elif "llama" in model_name:
                api_key = st.secrets["GROQ_API_KEY"]
        
        self.model_name = model_name
        
        if "gpt" in model_name:
            self.chatbot = ChatOpenAI(max_tokens=2048, openai_api_key=api_key, model_name=model_name)
        elif "llama" in model_name:
            self.chatbot = ChatGroq(temperature=0, api_key=api_key, model_name=model_name)

    def stream(self, messages):
        if "gpt" in self.model_name:
            messages = messages
        elif "llama" in self.model_name:
            messages = [{"role": get_role(message), "content": message.content[0]["text"]} for message in messages]
        
        for response in self.chatbot.stream(messages):
            yield response



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
        messages = [SystemMessage(content=[{ "type": "text", "text": st.session_state.system_message }])] + messages

    st.session_state.cost = st.session_state.get("cost", 0) + estimate_cost(messages)
    
    message = AIMessage(content=[full_response])
    st.session_state.messages.append(message)
    

    try:
        for response in st.session_state.chatbot.stream(messages):
            full_response["text"] += response.content
            message.content = [full_response]
            placeholder.markdown(full_response["text"] + "▌")
        placeholder.markdown(full_response["text"])
        st.session_state.cost = st.session_state.get("cost", 0) + estimate_cost([message]) * 3
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


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def estimate_cost(messages):
    cost = 0
    
    multiplier = model_costs_multiplier[st.session_state.chatbot.model_name] # Per 1M tokens
    
    for message in messages:
        if message.content[0]["type"] == "text":
            num_tokens = num_tokens_from_string(message.content[0]["text"], "cl100k_base")
            cost += num_tokens * multiplier * 1e-6
        elif message.content[0]["type"] == "image_url":
            cost += 300 * 1e-6
            
    return cost


def st_chatbot(api_key):
    
    model_name = st.sidebar.radio("Select model:", model_costs_multiplier.keys(), index=3)
    
    if (model_name != st.session_state.get("model_name", "")) or (api_key != st.session_state.get("api_key", "")):
        st.session_state.api_key = api_key
        st.session_state.model_name = model_name
        st.session_state.chatbot = ChatBot(api_key=api_key, model_name=model_name)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    with st.sidebar:
        st.text_area("System message", key="system_message", value="")
        st.session_state.chatbot.temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        st.button("Reset chat", on_click=lambda: st.session_state.pop("messages", None) and st.session_state.pop("cost", None), type="primary", use_container_width=True)
        regenerate = st.button("Regenerate response", on_click=lambda: st.session_state.messages.pop(), use_container_width=True)
        st.button("Delete last", on_click=lambda: st.session_state.messages.pop(), use_container_width=True)
        stop = st.button("Stop", use_container_width=True)

        if "gpt-4" in model_name:
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

    st.title(f"Chatbot UI  (*{model_name}*)")
    
    container = st.container()
    # container = st.container(height=550, border=None)
    
    with container:
        for message in st.session_state.messages:
            print_message(message)
            
    # React to user input
    if (prompt := st.chat_input(f"Write here...")) and not stop:
        message = HumanMessage(content=[{
                "type": "text",
                "text": prompt,
        }])
        st.session_state.messages.append(message)
        with container:
            print_message(message)
            generate_response()
    elif regenerate and not stop:
        with container:
            generate_response()
    
    st.markdown(f"*Cost: ${st.session_state.get('cost', 0):.5f}*")
