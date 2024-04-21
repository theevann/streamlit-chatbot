import streamlit as st

from chatbot import st_chatbot
from dall_e import st_dall_e
from password import check_password


st.set_page_config(
    page_title="ChatBot UI",
    layout="wide",
)


# if not check_password():
#     st.stop() 

api_key = st.sidebar.text_input("API Key")

mode = st.sidebar.radio("Select tool:", ["ChatBot", "DALL·E"])

st.sidebar.write("---")

if api_key == "":
    st.warning("Please enter your API key.")
    st.stop()
        
if mode == "ChatBot":
    st_chatbot(api_key)
else:
    st_dall_e(api_key)