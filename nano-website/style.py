import streamlit as st

def set_style():
    st.set_page_config(page_title="Nanoparticles", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
        }
        .header {
            background-color: #90E593;
            color: black;
            font-weight: bold;
            font-size: 3vw;
            text-align: center;
            padding: 20px;
        }
        .about
        {
            color: white;
            font-weight: bold;
            font-size: 1.5vw;
            padding-top: 10px;
            padding-bottom: 10px;
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
            width: 100%;
            height: 400px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
