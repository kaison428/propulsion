import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st
from streamlit_folium import folium_static

import pandas as pd
import numpy as np
import folium
import time
import os

from model import *

# Set API Key ----------------------------------------------------------------
os.environ["COHERE_API_KEY"] = st.secrets['COHERE_API_KEY']
os.environ["WEAVIATE_API_KEY"] = st.secrets['COHERE_API_KEY']
os.environ["WEAVIATE_URL"] = st.secrets['COHERE_API_KEY']
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']

# Set session state ---------------------------------------------------------
def clear_submit():
    st.session_state["submit"] = False

# configuration ---------------------------------------------------------
st.set_page_config(page_title="Propulsion | Proposal Co-pilot", page_icon=":building_construction:", layout='wide')
st.header("Propulsion - Accelerate Your Next Project")

# Sidebar contents ----------------------------------------------------------------
with st.sidebar:
    st.title(':building_construction: Propulsion')
    st.markdown('''
    ## About
    This app is a Proposal Co-pilot built using:
    - [Streamlit](https://streamlit.io/)
    - [Cohere](https://cohere.com/)
    - [Weaviate](https://weaviate.io/)
    - [LangChain](https://python.langchain.com/en/latest/)
    
    ''')

    input_container = st.container()

    # User input ------------------------------------------------------------------
    ## Function for taking user provided PDF (e.g. RFP) as input

    def get_file(key):
        uploaded_files = st.file_uploader(f"Upload your {key}", type='pdf', key=key, on_change=clear_submit, accept_multiple_files=True)
        return uploaded_files

    ## Applying the user input box
    with input_container:
        files = get_file('RFP / Proposal')
        process_button = st.button("Process")

        # Process RFP / document ----------------------------------------------------------------
        def summarize_data(files) -> str:
            # 1. Read and process data
            combined_text = '\n---\n'.join([get_text_from_pdf(f) for f in files])

            return get_summary(combined_text)

        if process_button:
            st.session_state.messages = []

            if not files:
                st.error("Please upload at least one document!")
            else:
                with st.spinner('Processing...'):
                    start_time = time.time()
                    st.session_state["summary"], st.session_state["title"] = summarize_data(files)
                    st.session_state["coordinates"] = get_coordinates(get_location(st.session_state["summary"]))
                    st.session_state["SWOT"] = get_swot_analysis(st.session_state["summary"])
                    print(time.time() - start_time)

                st.success('Done!')
                st.session_state["submit"] = True

    add_vertical_space(5)
    st.write('Made by Kaison Cheung')

# Layout of input/response containers ----------------------------------------------------------------
context_container = st.container()
colored_header(label='', description='', color_name='blue-30')
overview_tab, swot_tab = st.tabs(["Overview", "SWOT"])
chat_container = st.container()

# Tabs organization ----------------------------------------------------------------
with overview_tab:
    map_col, summary_col = st.columns([0.4, 0.6])

with swot_tab:
    swot_col, swot_ref_col = st.columns([0.5, 0.5])

# Map Display ------------------------------------------------------------------
with map_col:
    # center on Liberty Bell, add marker
    if st.session_state.get("coordinates"):

        location = "Approximate Site Location"
        if st.session_state["coordinates"] == [0,0]:
            location = 'Location Failure'

        m = folium.Map(location=st.session_state["coordinates"], zoom_start=16)
        folium.Marker(
            st.session_state["coordinates"], popup=location, tooltip=location
        ).add_to(m)

        # call to render Folium map in Streamlit
        st_data = folium_static(m, width=350)

    else:
        m = folium.Map(location=[43.65349167474285, -79.38440687827095], zoom_start=16)
        folium.Marker(
            [43.65349167474285, -79.38440687827095], popup="City of Toronto", tooltip="City of Toronto"
        ).add_to(m)

        # call to render Folium map in Streamlit
        st_data = folium_static(m, width=350)

# Summarys Display ------------------------------------------------------------------
with summary_col:
    if st.session_state.get("submit"):
       st.subheader(st.session_state["title"])
       st.markdown(st.session_state["summary"])


# # SWOT Analysis Display ------------------------------------------------------------------
# with swot_col:
#     if st.session_state.get("SWOT"):
#        st.subheader('SWOT Analysis')
#        st.markdown(st.session_state["SWOT"])

# Chatbot UI ----------------------------------------------------------------
with chat_container:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        role = 'assistant' if message["role"] == 'CHATBOT' else message["role"]
        with st.chat_message(role):
            st.markdown(message["message"])

    if prompt := st.chat_input("Ask me anything about the RFP or proposal!"):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if st.session_state.get('summary'):
                summary = st.session_state['summary']
                chat_history = [
                    {"role": "USER", "message": f'Answer the following questions with reference to this RFP summary if required: {summary}'}
                ] + st.session_state.messages
                full_response, raw_response = chat_from_database(prompt, chat_history=chat_history)
            else:
                full_response, raw_response = chat_from_database(prompt, chat_history=st.session_state.messages)

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "USER", "message": prompt})
        st.session_state.messages.append({"role": "CHATBOT", "message": raw_response})