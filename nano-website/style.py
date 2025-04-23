import streamlit as st

def set_style():
    st.set_page_config(page_title="Nanoparticles", layout="wide")
    st.markdown("""
    <style>

        h3 {
            font-size: 21px !important;
            padding: 8px 0px 5px 0px !important;
        }

        ::-webkit-scrollbar {
            display: none;
        }

	    .stTabs [data-baseweb="tab"] {
		    height: 40px;
		    border-radius: 4px 4px 0px 0px;
            white-space: pre-wrap;
            padding: 0px 10px 0px 10px;
        }

        .stTabs [data-baseweb="tab"] p {
            font-size: 1.0vw;            
        }

        .stTabs [data-baseweb="tab-panel"] {
            #border: medium solid rgba(49, 51, 63, 0.1);
            padding: 10px 10px 10px 10px;
		    border-radius: 0px 0px 4px 4px;            
        }

        .stExpander details {
            border-top: 1px dashed gray;
            border-right: 0px dashed gray;
            border-bottom: 1px dashed gray;
            border-left: 0px dashed gray;
        }

        .st-key-right_button button{
            float: right;
        }

        .text {
            font-size: 20px;
            text-align: justify;
        }

        .cite {
            font-size: 1.5vw;
        }

        .header {
            background-color: rgba(0,0,255,0.25);
            color: black;
            font-weight: bold;
            font-size: 3vw;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
        }

        .about
        {
            font-weight: bold;
            font-size: 2vw;
            text-align: center;
        }

        .footer {
            background-color: #f1f1f1;
            color: #808080;
            text-align: center;
            padding: 1px;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
        }

    </style>
    """, unsafe_allow_html = True)
