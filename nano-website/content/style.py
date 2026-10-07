import streamlit as st

def loadStyles(color):
    st.markdown(f"""
        <style>    
            h3 {{
                font-size: 21px !important;
                padding: 8px 0px 5px 0px !important;
            }}

            [data-testid = "stSidebar"] {{
                width: 350px;
            }}

            .stExpander details {{
                border-top: 1px dashed gray;
                border-right: 0px dashed gray;
                border-bottom: 1px dashed gray;
                border-left: 0px dashed gray;
            }}

            .st-key-button_contact button p {{
                font-size: 20px;
                color: black;
            }}

            .st-key-button_contact button {{
                background-color: rgba(255, 255, 255, 1);
            }}

            .text {{
                font-size: 20px;
                text-align: justify;
            }}

            .text-help {{
                font-size: 22px;            
            }}

            .text.center {{ text-align: center; }}

            .cite {{
                font-size: 21px;
                text-align: justify;
            }}

            .header {{
                background-color: {color};
                color: black;
                font-weight: bold;
                font-size: 45px;
                text-align: center;
                padding: 5px;
                border-radius: 10px;
            }}

            .about
            {{
                font-weight: bold;
                font-size: 33px;
                text-align: center;
            }}

            .footer {{
                background-color: rgb(0, 0, 0);
                color: #808080;
                text-align: center;
                padding: 1px;
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                z-index: 9999;
            }}

        </style>
    """, unsafe_allow_html = True)
