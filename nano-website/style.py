import streamlit as st

def set_style():
    st.set_page_config(page_title="Nanoparticles", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        .text {
            font-size: 20px;
        }
        .cite {
            font-size: 1.25vw;
            padding-top: 25px;
            padding-bottom: 25px;
            text-align: center;
        }
        .header {
            background-color: #90E593;
            color: black;
            font-weight: bold;
            font-size: 3vw;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            padding-bottom: 35px;
        }
        .about
        {
            font-weight: bold;
            font-size: 2vw;
            text-align: center;
        }
        .footer {
            background-color: #f1f1f1;
            color: #4CAF50;
            text-align: center;
            padding: 10px;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
        }
        .calculations {
            color: white;
            font-weight: bold;
            font-size: 1.5vw;
            padding-bottom: 10px;
        }
        .image-container {
            background: transparent;
            text-align: center;
        }
        .image-display {
            width: inherit;
            height: 400px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
