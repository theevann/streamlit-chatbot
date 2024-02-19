import streamlit as st

from chatbot import st_chatbot
from dall_e import st_dall_e
from password import check_password


st.set_page_config(
    page_title="OpenAI UI",
    layout="wide",
)


# if not check_password():
#     st.stop() 

openai_api_key = st.sidebar.text_input("OpenAI API Key")

mode = st.sidebar.radio("Select tool:", ["ChatGPT", "DALL·E"])

st.sidebar.write("---")


if mode == "ChatGPT":
    st_chatbot(openai_api_key)
else:
    st_dall_e(openai_api_key)