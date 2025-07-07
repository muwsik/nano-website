# -*- coding: cp1251 -*-

import streamlit as st
import requests
import uuid
import datetime
import io
import base64


# TOKEN = st.secrets["TOKEN"]
# CHAT_ID = st.secrets["CHAT_ID"]
# def message2telegram(_session_state, _traceback):
#     messageUID = str(uuid.uuid4())
#     successQ = False

#     message =   f"Time: {datetime.datetime.now().ctime()} \n" + \
#                 f"Dump: {str(_session_state)} \n" + \
#                 f"{_traceback.format_exc()} \n"

#     textResponse = requests.post(
#         url = f"https://api.telegram.org/bot{TOKEN}/sendDocument",
#         files = {"document": (f"{messageUID}.txt", io.BytesIO(message.encode()))},
#         data = {"chat_id": CHAT_ID, "caption": f"UID: {messageUID}"}
#     )

#     if textResponse.json().get("ok"):
#         successQ = True

#     imageResponse = None
#     if _session_state['uploadedImg'] is not None:
#         imageResponse = requests.post(
#             url = f"https://api.telegram.org/bot{TOKEN}/sendDocument",        
#             files = {"document": (f"{messageUID}-{_session_state['uploadedImg'].name}", _session_state['uploadedImg'].getvalue())},
#             data = {"chat_id": CHAT_ID, "caption": f"UID: {messageUID}"}
#         )    

#         if successQ and imageResponse.json().get("ok"):
#             successQ = True
#         else: successQ = False

#     return successQ, [textResponse, imageResponse]


URL = r"https://script.google.com/macros/s/AKfycbzygeiJmBaL4Qu9FRL2iS-bPrFGLPCFgyut72ZE6UHA9hr3ChbTkh1_7BwbNXRqxw9Y/exec"    
EMAIL = r"nanoweb.assist@gmail.com"

def message2email(data):
    messageUID = str(uuid.uuid4())

    payload = {
        "to": EMAIL,
        "subject": f"NanoWebsite {datetime.datetime.now().ctime()}",
        "body": 
            f"UID: {messageUID}\n\n" + \
            f"By: *response address*",
        "dumpData": f"Dump: {data['dump']} \n\n" + f"{data['traceback']}",
        "imageType": "None",
        "imageData": "None"
    }

    if (data["image-type"] is not None) and (data["image-data"] is not None):
        payload.update({
            "imageType": data["image-type"],
            "imageData": base64.b64encode(data["image-data"]).decode("utf-8")
        })

    response = requests.post(URL, json = payload)
    
    if response.json().get("ok"):
        return True, response

    return False, response


if __name__ == "__main__":
    
    message2email({
        "dump": "123",
        "traceback": "234",
        "image-type": r"image/tiff",
        "image-data": b'de1Faf'        
    })