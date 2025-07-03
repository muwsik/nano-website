import streamlit as st
import requests
import uuid
import datetime
import io


TOKEN = st.secrets["TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]
  

def send_message(_session_state, _traceback):
    messageUID = str(uuid.uuid4())
    successQ = False

    message =   f"Time: {datetime.datetime.now().ctime()} \n" + \
                f"Dump: {str(_session_state)} \n" + \
                f"{_traceback.format_exc()} \n"

    response_text = requests.post(
        url = f"https://api.telegram.org/bot{TOKEN}/sendDocument",
        files = {"document": (f"{messageUID}.txt", io.BytesIO(message.encode()))},
        data = {"chat_id": CHAT_ID, "caption": f"UID: {messageUID}"}
    )
    if response_text.json().get("ok"):
        successQ = True

    response_image = None
    if _session_state['uploadedImg'] is not None:
        response_image = requests.post(
            url = f"https://api.telegram.org/bot{TOKEN}/sendDocument",        
            files = {"document": (f"{messageUID}-{_session_state['uploadedImg'].name}", _session_state['uploadedImg'].getvalue())},
            data = {"chat_id": CHAT_ID, "caption": f"UID: {messageUID}"}
        )    

        if successQ and response_image.json().get("ok"):
            successQ = True
        else: successQ = False

    return successQ, [response_text, response_image]