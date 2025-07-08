# -*- coding: cp1251 -*-

import streamlit as st
import requests
import datetime
import uuid
import base64


PASSWORD = st.secrets["PASSWORD"]
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
        "imageData": "None",
        "secret": PASSWORD
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