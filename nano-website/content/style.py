import streamlit as st

def loadStyles(color):
    st.markdown(f"""
        <style>  
            iframe {{
                width: 100%;
                height: 90vh !important;                
                min-height: 350px;
                max-height: 750px;
            }}

            div.stVerticalBlock.st-key-image-container > div {{
                height: 100% !important;
            }}

            ::-webkit-scrollbar {{
                display: none;
            }}            

            h3 {{
                font-size: 21px !important;
                padding: 8px 0px 5px 0px !important;
            }}

	        .stTabs [data-baseweb = "tab"] {{
		        height: 40px;
		        border-radius: 4px 4px 0px 0px;
                border: medium solid rgba(200, 200, 200, 0.3);
                padding: 0px 10px 0px 10px;
            }}

            .stTabs [aria-selected = "false"] {{
                 background-color: rgba(233, 233, 233, 0.25);
            }}

            .stTabs [data-baseweb = "tab"] p {{
                font-size: 20px;            
            }}

            .stTabs [data-baseweb = "tab-panel"] {{
                padding: 10px 10px 10px 10px;
		        border-radius: 0px 0px 4px 4px;            
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
                background-color: rgb(230, 230, 255);
            }}

            .text {{
                font-size: 20px;
                text-align: justify;
            }}

            .text.center {{ text-align: center; }}

            .cite {{
                font-size: 1.25vw;
                text-align: justify;
            }}

            .header {{
                background-color: {color};
                color: black;
                font-weight: bold;
                font-size: 3vw;
                text-align: center;
                padding: 20px;
                border-radius: 10px;
            }}

            .about
            {{
                font-weight: bold;
                font-size: 2vw;
                text-align: center;
            }}

            .particle-label {{
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-weight: bold;
                display: inline-block;
                margin: 2px;
            }}

            .particle-label.blue    {{ background-color: #007bff; }}
            .particle-label.green   {{ background-color: #28a745; }}
            .particle-label.red     {{ background-color: #dc3545; }}
            .particle-label.orange  {{ background-color: #fd7e14; }}

            .footer {{
                background-color: rgb(0, 0, 0);
                color: #808080;
                text-align: center;
                padding: 1px;
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
            }}

        </style>
    """, unsafe_allow_html = True)
