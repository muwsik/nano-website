# Run application
# streamlit run .\nano-website\nano-website.py

from json import tool
import streamlit as st

import io, csv
import cv2, skimage, scipy
import numpy as np
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt

# content
import content.style as style
import content.tooltips as tooltips
import content.instructions as instruct

# utils
import utils.autoscale as autoscale
import utils.NanoStatistics as NanoStat
import utils.ExponentialApproximation as ExpApp
import utils.ExponentialApproximation2 as ExpApp2
import utils.CustomComponents as CustComp
import utils.WebsiteBot as webBot
import utils.API2CVAT as API2CVAT
import utils.accuracy as accuracy
import utils.structured as structured


import traceback
import joblib
    

### Function ###
    
colorRGBA_str = 'rgb(150, 150, 255)'
colorRGB = (75, 255, 75)


@st.cache_resource
def load_SVM_model():
    return joblib.load("./nano-website/content/svc_best-comb_mean2-3-new.joblib")


def defaultDetectTab():
    st.session_state['imgUpload'] = False
    st.session_state['uploadedImg'] = None
    st.session_state['fileImageName'] = None
    st.session_state['srcImg'] = None
    st.session_state['typeImg'] = None

    st.session_state['imgPlaceholder'] = None

    st.session_state['settingDefault'] = False
            
    st.session_state['detected'] = False
    st.session_state['BLOBs_data'] = None
    st.session_state['BLOBs_filter'] = None
    st.session_state['detectedParticles'] = 0    
    st.session_state['filteredParticles'] = 0 
    st.session_state['imgBLOB'] = None
    st.session_state['shapesBLOB'] = None
    st.session_state['timeDetection'] = None
    st.session_state['detectionSettings'] = None

    st.session_state['comparison'] = True
    st.session_state['displayScale'] = False
    st.session_state['areas'] = False
    st.session_state['big_contours'] = None


def defaultStatTab():
    st.session_state['calcStatictic'] = False
    
    st.session_state['statBLOBs'] = None
    st.session_state['statImage'] = None
    st.session_state['statImageName'] = None

    st.session_state['distView'] = False
    st.session_state['normalize'] = False
    st.session_state['selection'] = False
    st.session_state['step'] = 0.5

    st.session_state['struct'] = None


def loadDefault_sessionState(_dispToast = False):
    if _dispToast:
        st.toast('Default configuration loaded!')
    
    st.session_state['rerun'] = False
    
    st.session_state['sizeImage'] = None
    st.session_state['scale'] = None
    st.session_state['scaleData'] = None

    defaultDetectTab()
    defaultStatTab()


def sessionState2str(closedKey = ["imgPlaceholder", ]):
    tempStr = "\n"
    for key in st.session_state.keys():
        if key not in closedKey:
            tempStr = tempStr + f"\t{key}: {str(st.session_state[key])}\n"

    return tempStr


@st.dialog("Something went wrong...")
def dialog_exception(sendReportFlag = True):
    st.write("""
        An error occurred while the application was running.
        *The latest detection and marking results are saved.*
        Refresh the site with the "Rerun page" button below
        (if you refresh the page through the browser, some data may not be saved).
    """)

    if st.button("Rerun page", type = "primary"): 
        st.session_state['rerun'] = True
        st.rerun()
    else:
        dataException = {
            "dump": sessionState2str(),
            "contact-email": "None",
            "add-info": traceback.format_exc(),    
            "image-data": None,
            "image-type": None
        }

        if st.session_state['uploadedImg'] is not None:
            dataException.update({                
                "image-data": st.session_state['uploadedImg'].getvalue(),
                "image-type": st.session_state['uploadedImg'].type
            })       

        if sendReportFlag:
            result, response = webBot.message2email(dataException)


    with st.expander("Info for developers", expanded = False, icon = ":material/app_registration:"):
        st.error(traceback.format_exc())
        
        if sendReportFlag:
            if result:
                st.success("Report successful sent!")
            else:
                st.error("Error sending report: " + str(response.json()))
    
            
@st.dialog("Send feedback...")
def dialog_feedback():
    submitButtonClick = False

    with st.form(key = "feedback-form"):
        contactEmail = st.text_input("Your contact E-mail")

        txt = st.text_area("Describe your problem")

        sendImg = False
        if st.session_state["imgUpload"]:
            sendImg = st.toggle("The current uploaded image on site will be sent", value = True)

        submitButtonClick = st.form_submit_button('Send feedback', icon = ":material/drafts:")

    if submitButtonClick:
        dataFeedback = {
            "dump": sessionState2str(),
            "contact-email": contactEmail,
            "add-info": txt,    
            "image-data": None,
            "image-type": None
        }

        if sendImg:
            dataFeedback.update({                
                "image-data": st.session_state['uploadedImg'].getvalue(),
                "image-type": st.session_state['uploadedImg'].type
            })

        result, _ = webBot.message2email(dataFeedback)
        
        if result:
            st.success("Feedback successful sent!")
        else:
            st.error("Error sending feedback. Please try again...")


def update_sessionState(key, value):
    st.session_state[key] = value



### Main app ###
try:
    # Loading CSS styles
    st.set_page_config(page_title = "Web Nanoparticles", layout = "wide")
    style.loadStyles(colorRGBA_str)
    
    # Initial loading of session states
    if 'rerun' not in st.session_state:
        loadDefault_sessionState()
    elif st.session_state['rerun']:
        loadDefault_sessionState(True)
    
    ## Header
    instruct.Header()

    ## About
    instruct.About()

    ## Main content area
    tabDetect, tabStat, tabHelp = st.tabs([
        "Automatic detection",
        "Statistics dashboard",
        "Help"
    ])


    ## TAB 1
    with tabDetect:   
        imgPlaceholder = None
        
        st.subheader("Upload SEM image", anchor = False)            
        uploadedImg = st.file_uploader("Choose an SEM image", type = ["tif", "tiff", "png", "jpg", "jpeg" ])
        st.session_state['uploadedImg'] = uploadedImg

        if uploadedImg is not None: 
            if (st.session_state['fileImageName'] != uploadedImg.name):                
                srcImage = Image.open(uploadedImg).convert("L")
                
                srcImage = srcImage.resize((1280, 960)) # TO DO:fix

                defaultDetectTab()
                st.session_state['srcImg'] = srcImage
                st.session_state['fileImageName'] = uploadedImg.name
            else:
                srcImage = st.session_state['srcImg']   
                
            st.session_state['imgUpload'] = True       
        else:
            defaultDetectTab()
        
        
        if (st.session_state['imgUpload']):
            colImage, colSetting = st.columns([6, 2])

            with colImage:
                with st.container(key = 'image-container'):
                    if (st.session_state['imgPlaceholder'] is None):
                        st.session_state['imgPlaceholder'] = st.empty()

            # Detection settings and results
            with colSetting: 
                # Preprocessing image
                with st.spinner("Preprocessing image...", show_time = True):  
                    st.toggle("Use default settings",
                        disabled = not st.session_state['imgUpload'],
                        key = 'settingDefault',
                        help = tooltips.DefaultToggle
                    )                         
                    
                    tempArrImg = np.array(srcImage, dtype = 'uint8')

                    st.session_state['scale'], st.session_state['scaleData'] = autoscale.estimateScale(
                        tempArrImg
                    )

                    lowerBound = autoscale.findBorder(tempArrImg)                    
                    if (lowerBound is not None):
                        srcImage = srcImage.crop((0, 0, srcImage.size[0], lowerBound))
                        
                    st.session_state['sizeImage'] = srcImage.size

                    if (st.session_state['typeImg'] is None) or st.session_state['settingDefault']:
                        data = np.array(srcImage, dtype = 'uint8').flatten()
                        counts, _ = np.histogram(data, bins = np.arange(0, 255, 1))
                        counts = counts / np.sum(counts)

                        cumSum = 0
                        for i, j in enumerate(counts):
                            cumSum = cumSum + j
                            if (cumSum >= 0.5):
                                if (i <= 127):
                                    st.session_state['typeImg'] = 1     # SEM
                                else: st.session_state['typeImg'] = 2   # TEM
                                break
                            
                # Detection settings       
                with st.expander("Detection settings", expanded = not st.session_state['detected'], icon = ":material/tune:"):
                    st.radio("Type of microscope image",
                        key = 'typeImg',
                        options = tooltips.Options.TypeMicroscope.keys(),
                        format_func = lambda option: tooltips.Options.TypeMicroscope[option],                        
                        horizontal = True,                        
                        disabled = st.session_state['settingDefault'],
                        help = tooltips.TypeMicroscopePills
                    )                    
                    
                    if ('param-pre-1' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-1'] = 5
                    
                    st.slider("Minimal nanoparticle brightness",
                        key = 'param-pre-1',
                        disabled = st.session_state['settingDefault'],
                        help = tooltips.Detection.Brightness
                    )

                    if ('param-pre-2' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-2'] = 2

                    st.selectbox("Hypothetical nanoparticles diameter",
                        key = 'param-pre-2',
                        options = tooltips.Options.NanopartSize.keys(),
                        format_func = lambda option: tooltips.Options.NanopartSize[option],
                        disabled = st.session_state['settingDefault'],
                        help = tooltips.Detection.Diameter
                    )

                    if ('param-pre-3' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-3'] = False

                    st.toggle("Suppression of background irregularities",
                        key = 'param-pre-3',                              
                        disabled = st.session_state['settingDefault'],                        
                        help = tooltips.Detection.Irregularities
                    )
        
                    pushDetectButton = st.button("Nanoparticles detection",
                        use_container_width = True,
                        disabled = not st.session_state['imgUpload'],
                        on_click = update_sessionState,
                        args = ("detected", True)
                    )
                    
                    tempWarningPlaceholder = st.empty()
                    tempSettings = [                        
                        st.session_state['typeImg'],
                        st.session_state['param-pre-1'],
                        st.session_state['param-pre-2'],
                        st.session_state['param-pre-3']
                    ]
                    if (st.session_state['detectionSettings'] is not None) and st.session_state['detected']:
                        if (st.session_state['detectionSettings'] != tempSettings):
                            tempWarningPlaceholder.warning(tooltips.Warnings.DetectSettings,
                                icon = ":material/warning:"
                            )
                
                # Detecting
                if pushDetectButton:
                    st.session_state['detected'] = False
                    tempWarningPlaceholder.empty()
                    
                    timeStart = time.time()
                    with st.spinner("Nanoparticles detection...", show_time = True): 
                        
                        if (st.session_state['typeImg'] == 2):
                            srcImage = ImageOps.invert(srcImage)

                        currentImage = np.array(srcImage, dtype = 'uint8') 
                        
                        lowerBound = autoscale.findBorder(currentImage)        
                        if (lowerBound is not None):
                            currentImage = currentImage[:lowerBound, :]

                        if st.session_state['param-pre-2'] == 0:
                            # параметры в пикселях
                            params = {
                                # размер окна медианного фильтра
                                "sz_med" : 3,
                                # параметр функции Гаусса для сглаживания
                                # "sigma_gauss": -,
                                # размер диска Top-Hat
                                "sz_th":  4,
                                # порог яркости для отбрасывания лок. максимумов
                                "thr_br": float(st.session_state['param-pre-1']),   
                                # минимальное расстояние между локальными максимумами при их поиске 
                                "min_dist": 3,
                                # размер окна аппроксимации
                                "wsize": 7,     
                                # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                                "best_mode": 3, 
                                # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                                "msk": 3,      
                                # аппроксимирующая функция "exp" или "pol" 
                                "met": 'exp',   
                                # число параметров аппроксимации
                                "npar": 2,
                                # удаление сложных засветов
                                "deleteBorderLines": st.session_state['param-pre-3'], 
                                # порог бинаризации для обнаружения сложных засветов
                                "threshLines": 100,
                                # потенциальное количество наночастиц
                                "nlocmax": 5000,
                            }
                                                    
                            currentImage = ExpApp.PreprocessingMedian(currentImage, params['sz_med'])
                            currentImage = ExpApp.PreprocessingTopHat(currentImage, params['sz_th'])                            

                        elif st.session_state['param-pre-2'] == 1:
                            # параметры в пикселях
                            params = {
                                # размер окна медианного фильтра
                                "sz_med" : 3,
                                # параметр функции Гаусса для сглаживания
                                "sigma_gauss": 1,
                                # размер диска Top-Hat
                                "sz_th":  6,
                                # порог яркости для отбрасывания лок. максимумов
                                "thr_br": float(st.session_state['param-pre-1']),   
                                # минимальное расстояние между локальными максимумами при их поиске 
                                "min_dist": 3,
                                # размер окна аппроксимации
                                "wsize": 9,     
                                # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                                "best_mode": 3, 
                                # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                                "msk": 3,      
                                # аппроксимирующая функция "exp" или "pol" 
                                "met": 'exp',   
                                # число параметров аппроксимации
                                "npar": 2,
                                # удаление сложных засветов
                                "deleteBorderLines": st.session_state['param-pre-3'], 
                                # порог бинаризации для обнаружения сложных засветов
                                "threshLines": 100,
                                # потенциальное количество наночастиц
                                "nlocmax": 1500,
                            }

                            currentImage = ExpApp2.PreprocessingMedian(currentImage, params["sz_med"])
                            currentImage = ExpApp2.PreprocessingTopHat(currentImage, params["sz_th"])                             
                            currentImage = scipy.ndimage.gaussian_filter(
                                currentImage,
                                sigma = params["sigma_gauss"]
                            )

                        elif st.session_state['param-pre-2'] == 2:
                            # параметры в пикселях
                            params = {
                                # размер окна медианного фильтра
                                "sz_med" : 3,
                                # параметр функции Гаусса для сглаживания
                                "sigma_gauss": 1.5,
                                # размер диска Top-Hat
                                "sz_th":  8,
                                # порог яркости для отбрасывания лок. максимумов
                                "thr_br": float(st.session_state['param-pre-1']),   
                                # минимальное расстояние между локальными максимумами при их поиске 
                                "min_dist": 3,
                                # размер окна аппроксимации
                                "wsize": 7,     
                                # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                                "best_mode": 3, 
                                # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                                "msk": 3,      
                                # аппроксимирующая функция "exp" или "pol" 
                                "met": 'exp',   
                                # число параметров аппроксимации
                                "npar": 2,
                                # удаление сложных засветов
                                "deleteBorderLines": st.session_state['param-pre-3'], 
                                # порог бинаризации для обнаружения сложных засветов
                                "threshLines": 100,
                                # потенциальное количество наночастиц
                                "nlocmax": 700,
                            }

                            currentImage = ExpApp2.PreprocessingMedian(currentImage, params["sz_med"])
                            currentImage = ExpApp2.PreprocessingTopHat(currentImage, params["sz_th"]) 
                            currentImage = scipy.ndimage.gaussian_filter(
                                currentImage[::2, ::2],
                                sigma = params["sigma_gauss"]
                            )

                        else:
                            raise ValueError("!")


                        nlocmax = params["nlocmax"]
                        numpeaks = max(1000, nlocmax)
                        lms = skimage.feature.peak_local_max(currentImage,
                            min_distance = params["min_dist"],
                            threshold_abs = params["thr_br"],
                            threshold_rel = None,
                            footprint = None,
                            labels = None,
                            num_peaks = numpeaks
                        )
                        lmblobs = lms[:nlocmax]
                                  
                        blobs_appr = np.array(ExpApp2.ApproximationMain(currentImage, lmblobs, params, 3, True))

                        if (st.session_state['param-pre-2'] == 0) or (st.session_state['param-pre-2'] == 1):
                            pass
                        elif st.session_state['param-pre-2'] == 2:                            
                            blobs_appr[:, :3] = blobs_appr[:, :3] * 2
                        else:
                            raise ValueError("!")

                        blobs_appr[:, 2] = blobs_appr[:, 2] * 2         # radius -> diametr
                        blobs_appr[:, [5, 6]] = blobs_appr[:, [6, 5]]   # swap params

                        st.session_state['detected'] = True              
                        st.session_state['BLOBs_data'] = blobs_appr
                        st.session_state['detectedParticles'] = blobs_appr.shape[0]
                        st.session_state['detectionSettings'] = [
                            st.session_state['typeImg'],
                            st.session_state['param-pre-1'],
                            st.session_state['param-pre-2'],
                            st.session_state['param-pre-3']
                        ]
        
                    st.session_state['timeDetection'] = int(np.ceil(time.time() - timeStart))

                # Detection results
                if st.session_state['detected']:
                    instruct.DetectResult(st.session_state['detectedParticles'], st.session_state['timeDetection'])

                    # Warning about not correctly detection results 
                    if (st.session_state['detectedParticles'] < 1):            
                        st.warning(tooltips.Warnings.NoFoundNanos, icon = ":material/warning:")
                                    
                # Action with correctly detection results
                if (st.session_state['detected'] and st.session_state['detectedParticles'] > 0):
                    # Filtration settings
                    with st.expander("Filtration settings", expanded = True, icon = ":material/filter_alt:"):
                        if ('param-filt-1' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-1'] = 7
                                                    
                        st.slider("Nanoparticle center brightness",
                            key = 'param-filt-1',
                            disabled = st.session_state['settingDefault'],
                            help = tooltips.Filtarion.Brightness
                        )

                        temp_max_r_nm = 10.0
                        temp_min_r_nm = 1.0
                        temp_r_step = 0.1
                        if (st.session_state['BLOBs_data'] is not None) and (st.session_state['scale'] is not None):
                            temp_max_r_nm = np.max(st.session_state['BLOBs_data'][:, 2]) * st.session_state['scale']
                            temp_min_r_nm = np.min(st.session_state['BLOBs_data'][:, 2]) * st.session_state['scale']
                            temp_r_step = (temp_max_r_nm - temp_min_r_nm) / 100
                            if temp_r_step < 0.2:
                                temp_r_step = 0.1
                            elif temp_r_step < 0.5:
                                temp_r_step = 0.5                                
                            else:
                               temp_r_step = 1.0

                        temp_max = np.ceil(temp_max_r_nm+1)
                        temp_min = max(np.floor(temp_min_r_nm-1), 1.0)

                        if ('param-filt-2' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-2'] = (temp_min, temp_max)

                        st.slider("Nanoparticle diameter, nm",
                            key = 'param-filt-2',                         
                            min_value = temp_min,
                            step = temp_r_step,
                            max_value = temp_max,
                            format = "%0.1f",
                            disabled = st.session_state['settingDefault'],
                            help = tooltips.Filtarion.Diameter
                        )

                        if ('param-filt-3' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-3'] = 0.75

                        st.slider("Nanoparticle reliability",
                            key = 'param-filt-3',
                            min_value = 0.0,
                            step = 0.01,
                            max_value = 1.0,
                            disabled = st.session_state['settingDefault'],
                            help = tooltips.Filtarion.Reliability
                        )
                        

                        if st.session_state['param-pre-3']:
                            if ('param-filt-4' not in st.session_state) or st.session_state['settingDefault']:
                                st.session_state['param-filt-4'] = 2500

                            st.slider("Area of background irregularities",
                                key = 'param-filt-4',
                                min_value = 0,
                                step = 25,
                                max_value = 5000,
                                disabled = st.session_state['settingDefault'],
                                help = tooltips.Filtarion.Irregularities
                            )

                            temp_img = ExpApp.PreprocessingMedian(st.session_state['srcImg'].copy(), 3)
                            temp_img = ExpApp.PreprocessingTopHat(temp_img, 9)   

                            _, img_contours, st.session_state['big_contours'] = ExpApp2.FindAreasToDelete(temp_img, 85, st.session_state['param-filt-4'])
                            temp_BLOBs_data = ExpApp2.DeleteBorderPointsM(st.session_state['BLOBs_data'], img_contours, 255)
                         
                    divider = 1
                    if st.session_state['scale'] is not None:
                        divider = st.session_state['scale']

                    params_filter = {
                        "thr_c0": st.session_state['param-filt-1'],
                        "min_thr_d": st.session_state['param-filt-2'][0] / divider,   
                        "max_thr_d": st.session_state['param-filt-2'][1] / divider, 
                        "thr_error": 1 - st.session_state['param-filt-3'], 
                    }
            
                    # Filtering
                    BLOBs_data_filt = ExpApp.my_FilterBlobs_change(
                        st.session_state['BLOBs_data'] if not st.session_state['param-pre-3'] else temp_BLOBs_data,
                        params_filter
                    )

                    if (BLOBs_data_filt.shape[0] != 0):
                        st.session_state['BLOBs_filter'] = BLOBs_data_filt[:, :3]
                        st.session_state['filteredParticles'] = st.session_state['BLOBs_filter'].shape[0]
                    else:
                        st.session_state['BLOBs_filter'] = []
                        st.session_state['filteredParticles'] = 0

                    if (st.session_state['filteredParticles'] < 1):
                        st.warning(tooltips.Warnings.FiltrSettings, icon = ":material/warning:")
                                      
                    # Info about filtered nanoparticles
                    instruct.FiltrationResult(st.session_state['filteredParticles'])
                                        
                    with st.expander("Visualization and saving results", expanded = False, icon = ":material/display_settings:"):
                        # Displaying the scale
                        st.toggle("Estimated scale", key = 'displayScale', help = tooltips.Visualization.Scale)

                        if (st.session_state['displayScale'] and st.session_state['scaleData'] is None):
                            st.warning(tooltips.Warnings.OutScale, icon = ":material/warning:")                    

                        # Highlighting background irregularities
                        st.toggle("Highlighting background irregularities",
                            key = 'areas',
                            help = tooltips.Visualization.Irregularities
                        )
                            
                        # Saving
                        selectboxCol, buttonCol = st.columns([6,1], vertical_alignment = 'bottom')

                        selectionSave = selectboxCol.selectbox(
                            "What results should be saved?",
                            index = 3,
                            placeholder = "Select options...",
                            options = tooltips.Options.Saving.keys(),
                            format_func = lambda option: tooltips.Options.Saving[option]
                        )

                        fileResult = io.BytesIO()
                        fileResultName = 'None'
                        button_download_disabled = False

                        match selectionSave:
                            case 0:
                                temp = Image.new(mode = "RGBA", size = st.session_state['sizeImage'])
                                draw = ImageDraw.Draw(temp)
                                for BLOB in st.session_state['BLOBs_filter']:                
                                    y, x, d = BLOB; r = d/2          
                                    draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                temp.save(fileResult, format = 'png')
                                fileResultName = f"particls-{Path(uploadedImg.name).stem}.tif"

                            case 1:
                                imgBLOB = st.session_state['srcImg'].convert("RGB")
                                draw = ImageDraw.Draw(imgBLOB)                            
                                for BLOB in st.session_state['BLOBs_filter']:                
                                    y, x, d = BLOB; r = d/2
                                    draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                imgBLOB.save(fileResult, format = 'png')
                                fileResultName = f"particls+image-{Path(uploadedImg.name).stem}.tif"

                            case 2:
                                fileResult = io.StringIO()

                                temp_writer = csv.writer(fileResult, delimiter = ';')
                                if st.session_state['scale'] is not None: 
                                    temp_writer.writerow(["Scalse:", f"{st.session_state['scale']:.3}", "nm/px"])
                                else:
                                    temp_writer.writerow([f"Using default scale:", "1.0", "nm/px"])
                    
                                temp_writer.writerow(['coord y, px', 'coord x, px', 'diameters, px'])
                                temp_writer.writerows(st.session_state['BLOBs_filter'])

                                fileResultName = f"particls_info-{Path(uploadedImg.name).stem}.csv"
                            case 3:

                                st.write()

                                imageData= {
                                    'name': Path(uploadedImg.name).stem,
                                    'width': st.session_state['srcImg'].size[0],
                                    'height': st.session_state['srcImg'].size[1],
                                    'buffer': uploadedImg.getvalue()
                                }

                                fileResult = API2CVAT.ExportToCVAT(imageData, st.session_state['BLOBs_filter'])

                                fileResultName = f"{Path(uploadedImg.name).stem}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.zip"
                            case _:
                                button_download_disabled = True


                        buttonCol.download_button(
                            label = "",
                            icon = ":material/download:",
                            data = fileResult.getvalue(),
                            file_name = fileResultName,
                            disabled = button_download_disabled,
                            help = tooltips.Visualization.Download
                        )
    
        # Display source image by st.image
        if (st.session_state['imgUpload']):
            viewImage = st.session_state['srcImg'].copy().convert('RGB')
            draw = ImageDraw.Draw(viewImage)

            if (st.session_state['displayScale'] and (st.session_state['scaleData'] is not None)):
                y, x, length, text_scale = st.session_state['scaleData']
                diff_line = 5 # vertical line size
                y = y + 10 # vertical line shift
                x = x + 2 # horizontal line shift

                # Line of metric scale
                scaleLineCoords = [
                    (x,         y-diff_line),
                    (x,         y+diff_line),
                    (x,         y),
                    (x+length,  y),
                    (x+length,  y+diff_line),
                    (x+length,  y-diff_line)
                ]
                draw.line(scaleLineCoords, fill = colorRGB, width = 3)
                
                # Text of metric scale
                draw.text(
                    (x + int(length/8), y + 10), 
                    f"{length}px / {text_scale}",
                    fill = colorRGB,
                    font = ImageFont.load_default(size = 30) 
                )

            if ((st.session_state['filteredParticles'] > 0) and st.session_state['detected']):
                for BLOB in st.session_state['BLOBs_filter']:                
                    y, x, d = BLOB; r = d/2
                    draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)
            
            if st.session_state['areas']:
                if st.session_state['big_contours'] is not None:
                    viewImage = np.array(viewImage)
                    cv2.drawContours(
                        viewImage,
                        st.session_state['big_contours'],
                        thickness = -1, color = (255, 50, 50), contourIdx = -1
                    )
                    viewImage = Image.fromarray(viewImage)


            st.session_state['imgBLOB'] = viewImage
            
            if (st.session_state['comparison']):                
                CustComp.img_box(
                    st.session_state['srcImg'],
                    st.session_state['imgBLOB'],                        
                    st.session_state['imgPlaceholder']
                )
            else:
                CustComp.img_box(
                    st.session_state['imgBLOB'],
                    st.session_state['imgBLOB'],                        
                    st.session_state['imgPlaceholder']
                )


    ## TAB 2 
    with tabStat:    
        heightCol = 550
        marginChart = dict(l=10, r=10, t=40, b=5)
        marginChartLess = dict(l=5, r=5, t=0, b=5)
              
        with st.expander("Global dashboard settings",
            expanded = True,
            icon = ":material/rule_settings:"
        ):

            selectionUseNano = st.selectbox(
                "Which nanoparticles to use?",
                index = 2,
                options = tooltips.Options.NanoStatistic.keys(),
                format_func = lambda option: tooltips.Options.NanoStatistic[option],
                help = tooltips.NanopartSelectbox
            ) 
                
            match selectionUseNano:
                case 0:
                    st.session_state['calcStatictic'] = False
                    if (not st.session_state['detected']):
                        st.warning(tooltips.Warnings.NoResults, icon = ":material/warning:")
                    elif (st.session_state['filteredParticles'] < 10):
                        st.warning(tooltips.Warnings.SmallResults, icon = ":material/warning:")
                    else:                        
                        st.session_state['calcStatictic'] = True
                        st.session_state['statBLOBs'] = st.session_state['BLOBs_filter']
                        st.session_state['statImageName'] = Path(uploadedImg.name).stem
                        st.session_state['statImage'] = st.session_state['srcImg'].convert('RGB')
                case 1:
                    instruct.LabelUploderFileCVAT()
                    uploadedFileCVAT = st.file_uploader(
                        label = "Uploder CVAT file",
                        type = ["zip"],
                        label_visibility = 'collapsed')

                    if uploadedFileCVAT is None:
                        st.session_state['calcStatictic'] = False
                    else:
                        st.session_state['calcStatictic'] = True
                        st.session_state['statBLOBs'], st.session_state['statImageName'], imageCVAT = API2CVAT.ImportTaskFromCVAT(uploadedFileCVAT) 
                        
                        st.session_state['statImage'] = Image.open(imageCVAT).convert('RGB')

                        #TO DO: fix 
                        st.session_state['statImage'] =  st.session_state['statImage'].resize((1280, 960))

                        st.session_state['scale'], st.session_state['scaleData'] = autoscale.estimateScale(st.session_state['statImage'].convert("L"))
                        
                        if st.session_state['scale'] is not None:
                            st.session_state['sizeImage'] = (st.session_state['statImage'].size[0], st.session_state['scaleData'][0])
                        else:
                            st.session_state['sizeImage'] = st.session_state['statImage'].size
                case _:
                    st.session_state['calcStatictic'] = False

        if (not st.session_state['calcStatictic']):
            defaultStatTab()
        else:
            with st.expander("Particle parameters", expanded = True, icon = ":material/app_registration:"):
                instruct.AboutSectionParticleParams()
                                
                boolIndexSelectedBLOBs = None   

                diameter_nm = st.session_state['statBLOBs'][:, 2]
                BLOBs_nm = st.session_state['statBLOBs']
                if st.session_state['scale'] is not None:
                    diameter_nm = diameter_nm * st.session_state['scale']
                    BLOBs_nm = BLOBs_nm * st.session_state['scale']

                db11, db12, db13 = st.columns([4, 4, 4])            

                # Nanoparticle parameters
                with db12.container(border = True, height = heightCol):                        
                    
                    left, rigth = st.columns([7, 1])
                    left.subheader("Nanoparticle parameters", anchor = False)

                    with rigth.popover("", icon=":material/settings:"):
                        selectionDensity = st.selectbox(
                            "Particles material",
                            index = 0,
                            placeholder = "Select material type...",
                            options = tooltips.Options.MaterialDensity.keys(),
                            format_func = lambda option: tooltips.Options.MaterialDensity[option],
                        )

                        match selectionDensity:
                            case 0: materialDensity = 12.02 * 10**-12
                            case 1: materialDensity = 8.96 * 10**-12
                            case 2: materialDensity = 14.10 * 10**-12
                            case 3: materialDensity = 8.42 * 10**-12
                            case 4: pass;
                            case _: materialDensity = 1
                        
                        if selectionDensity == 4:   # User material
                            materialDensity = st.number_input(
                                "Particles material density on ng/nm³",
                                min_value = 0.0,
                                step = 1.0e-11,
                                value = 1.0e-10,
                                format = "%0.2e",
                                key = "user-density"
                            )
                        else:
                            instruct.MaterialDensity(materialDensity)
                    
                    # Additional info
                    if st.session_state['scale'] is None:
                        pass # TODO input scale
                    else:                    
                        instruct.EstimatedScale(st.session_state["scale"])
                    
                    if selectionDensity == 4: # User material
                        instruct.UserMaterial(tooltips.Options.MaterialDensity[selectionDensity], materialDensity)
                    else:
                        instruct.DefMaterial(tooltips.Options.MaterialDensity[selectionDensity])
                    
                    currentDiameter = diameter_nm
                    currentBLOBs = BLOBs_nm
                    if boolIndexSelectedBLOBs is not None:
                        currentDiameter = currentDiameter[boolIndexSelectedBLOBs]
                    
                    paramsNP = NanoStat.calculateParametersNP(
                        currentDiameter,
                        materialDensity,
                        st.session_state['sizeImage'],
                        st.session_state['scale'] # TODO add st.session_state with key 'areaImage'
                    )

                    # TODO fix boolIndexSelectedBLOBs
                    instruct.Quantity(len(st.session_state['statBLOBs']), len(currentDiameter))

                    # Primary parameters info                  
                    instruct.PrimaryParameters(currentDiameter)
                    
                    # Secondary parameters info 
                    instruct.SecondaryParameters(paramsNP)                   
                    
                    # Norm secondary parameters info
                    instruct.NormSecondaryParameters(paramsNP)                           
                # END db12
                

                # Particle size distribution
                with db11.container(border = True, height = heightCol):
                    left, rigth = st.columns([7, 1])
                    left.subheader("Distribution of particle diameters", anchor = False)

                    with rigth.popover("", icon=":material/settings:"):
                        st.toggle("Display distribution function",
                            key = 'distView',
                            help = tooltips.Distribution.Function
                        )

                        st.toggle("Normalize the vertical axis",
                            key = 'normalize',
                            help = tooltips.Distribution.Normalize
                        )

                        st.toggle("Selecting individual columns",
                            key = 'selection',
                            help = tooltips.Distribution.Selection
                        )

                        st.number_input("Histogram step",
                            key = 'step',
                            min_value = 0.1,
                            max_value = 5.0,
                            step = 0.1,
                            format = '%0.2f',
                            value = 0.5,
                            help = tooltips.Distribution.Step
                        )

                        # Saving data for distribution NP diams chart
                        buttonDataChartPlaceholder = st.empty()

                        # Saving chart distribution of NP diams 
                        buttonChartPlaceholder = st.empty()
                          
                    
                    step = st.session_state['step']
                    start = np.floor(diameter_nm.min()) - step
                    end = np.ceil(diameter_nm.max()) + step

                    counts, bins = np.histogram(diameter_nm, bins = np.arange(start, end, step, dtype = float))
                                        
                    name_x = "Diameters, nm"
                    temp = [[float(i), float(i+step)] for i in bins]
                    fraction = counts / np.sum(counts) * 100
                    if st.session_state['normalize']:
                        bar_y = fraction
                        name_y = "Particles fraction, %"
                        hover_y = "%{y:.1f}% (%{customdata[2]:d})"
                        dataChart = [list(pair) for pair in zip(temp, fraction)]
                        customDataChart = list(zip(bins, bins + step, counts))
                    else:
                        bar_y = counts
                        name_y = "Particles counts"
                        hover_y = "%{y:d} (%{customdata[2]:.1f}%)"
                        dataChart = [list(pair) for pair in zip(temp, counts)]
                        customDataChart = list(zip(bins, bins + step, fraction))
                                            
                    file = io.StringIO()
                    csv.writer(file, delimiter = ';').writerow([name_x, name_y])
                    csv.writer(file, delimiter = ';').writerows(dataChart)
                                        
                    buttonDataChartPlaceholder.download_button(
                        label = "Download data chart *.csv",
                        data = file.getvalue(),
                        file_name = f"{st.session_state['statImageName']}-dist-diameters.csv",
                        use_container_width  = True,
                        help = tooltips.Distribution.Download
                    )

                    fig = go.Figure()

                    fig = fig.add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = bar_y,
                        customdata = customDataChart,
                        showlegend = False,
                        hovertemplate = (
                            "Diameter: [%{customdata[0]:.1f}, %{customdata[1]:.1f}) nm<br>"
                            "Particls: " + hover_y +
                            "<extra></extra>"
                        )
                    ))

                    if st.session_state['distView']:
                        mu = np.mean(diameter_nm)
                        sigma = np.std(diameter_nm)

                        dist_x = np.arange(start, end, step * 0.1, dtype = float)
                        dist_y = np.exp(-1/2 * ((dist_x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

                        fig.add_trace(go.Scatter(
                            x = dist_x, 
                            y = dist_y * step * (100 if st.session_state['normalize'] else len(diameter_nm)),
                            mode = 'lines',
                            hoverinfo = 'skip',
                            showlegend = False,
                            line = dict(color = 'rgba(50, 50, 255, 0.75)')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x = [None], 
                            y = [None],
                            mode = 'lines',
                            line = dict(width = 0),    
                            showlegend = True,
                            name = f"Particles: {len(st.session_state['statBLOBs'])}<br>"
                                + f"Avg. diameter: {np.mean(diameter_nm):0.2f} nm<br>"
                                + f"Std. dev. diameter: {np.std(diameter_nm):0.1f} nm" 
                        )) 
                        
                    fig.update_layout(
                        margin = marginChartLess,
                        xaxis_title_text = name_x,
                        yaxis_title_text = name_y,                        
                        bargap = 0,
                        legend = dict(
                            x = 1.1,
                            y = 1.1,
                            xanchor = 'right',
                            yanchor = 'top',
                            bgcolor='rgba(0,0,0,0)'
                        )
                    )

                    fig.update_xaxes(
                        tickmode = 'linear',
                        dtick = 1,
                        tick0 = start,
                        tickwidth = 2,
                        showgrid = True,
                        gridwidth = 1,
                        minor = dict(
                            dtick = step,
                            ticklen = 4,
                            showgrid = False
                        )
                    )
                    
                    fig.update_traces(
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 1  
                    )

                    selectColumn = st.plotly_chart(
                        fig,
                        use_container_width = True,
                        on_select = 'rerun' if st.session_state['selection'] else 'ignore',
                        selection_mode = 'points'
                    )

                    if (st.session_state['selection']):
                        if (selectColumn.selection['point_indices'] != []):
                            minDiameterInColumn = selectColumn.selection['point_indices'][0] * step + start
                            maxDiameterInColumn = minDiameterInColumn + step

                            boolIndexSelectedBLOBs = (diameter_nm >= minDiameterInColumn) & (diameter_nm <= maxDiameterInColumn)
                # END db11

                
                # Heatmap of particle count
                # or
                # Visualization particles
                with db13.container(border = True, height = heightCol):
                    tempSelectionChart = st.selectbox(
                        "Type chart",
                        index = 1,
                        options = tooltips.Options.TypeChart.keys(),
                        format_func = lambda option: tooltips.Options.TypeChart[option],
                        label_visibility = 'collapsed'
                    )

                    match tempSelectionChart:
                        case 0: 
                            currentBLOBs = st.session_state['statBLOBs']
                            if boolIndexSelectedBLOBs is not None:         
                                currentBLOBs = currentBLOBs[boolIndexSelectedBLOBs]

                            stepSize = 10
                            uniformityMap = NanoStat.uniformity(
                                currentBLOBs,
                                st.session_state['sizeImage'],
                                stepSize
                            )

                            fig = px.imshow(uniformityMap, aspect = "equal")

                            fig.update_traces(
                                hoverinfo = "z",
                                hovertemplate = "Particle in subarea %{z:.2}<extra></extra>"
                            )

                            fig.update_layout(
                                margin = marginChartLess,
                                xaxis_title_text = f'Width image, {stepSize}*px',
                                yaxis_title_text = f'Height image, {stepSize}*px',
                                coloraxis_colorbar = dict(
                                    title = "Particle count",
                                    orientation = "h",
                                    y = -0.2,
                                ),
                                showlegend = False
                            )

                            st.plotly_chart(fig, use_container_width = True)
                        case 1: 
                            currentBLOBs = st.session_state['statBLOBs']
                            if boolIndexSelectedBLOBs is not None:         
                                currentBLOBs = currentBLOBs[boolIndexSelectedBLOBs]

                            tempImage = st.session_state['statImage'].copy()
                            draw = ImageDraw.Draw(tempImage)                            
                            for BLOB in currentBLOBs:                
                                y, x, d = BLOB; r = d/2
                                draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)
                            
                            st.image(tempImage, use_container_width = True)
                        case _:
                            pass

                    
                # END db13

            with st.expander("Nanoparticle spatial distribution", icon = ":material/data_thresholding:"):
                instruct.AboutSectionSpatialDistribution()

                currentBLOBs = np.copy(st.session_state['statBLOBs'])
                if st.session_state['scale'] is not None:         
                    currentBLOBs = currentBLOBs * st.session_state['scale']

                fullDist, minDist = NanoStat.euclideanDistance(currentBLOBs) 

                db21, db22, db23 = st.columns([1, 1, 1])
                
                # Fraction of empty subareas
                with db21.container(border = True, height = heightCol - 75):              
                    x = np.arange(5, 100, 5)

                    emptySubareas = np.zeros_like(x, dtype = 'float')

                    for i, size in enumerate(x):
                        temp = NanoStat.uniformity(
                            st.session_state['statBLOBs'],
                            st.session_state['sizeImage'],
                            size
                        )
                        emptySubareas[i] = np.sum(temp == 0) / (len(temp) * len(temp[0]))
                    
                    if (st.session_state['scale'] is not None):
                        fig = px.bar(x = x * st.session_state['scale'], y = emptySubareas)
                    else:
                        fig = px.bar(x = x, y = emptySubareas)

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Fraction of empty subareas", font = dict(size=27)),
                        xaxis_title_text = 'Size of square subareas, nm',
                        yaxis_title_text = 'Fraction of empty subareas',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hoverinfo = "x+y",
                        hovertemplate = "Size: %{x:.2} nm <br>Empty: %{y:.2}",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 1
                    )

                    st.plotly_chart(fig, use_container_width = True)

                # END db21

                # Distance to nearest nanoparticle
                with db22.container(border = True, height = heightCol - 75):                

                    fig = ff.create_distplot(
                        [minDist], [''], bin_size = 1, histnorm = 'probability',
                        colors = ['blue'], show_curve = False, show_rug = False
                    )

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Distance to nearest nanoparticle", font = dict(size=27)),
                        xaxis_title_text = 'Distance to nearest nanoparticle, nm',
                        yaxis_title_text = 'Particle fraction',
                        showlegend = False
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hoverinfo = "x",
                        hovertemplate = "Distanse: %{x:.2} nm",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 1,  
                    )

                    st.plotly_chart(fig, use_container_width = True)
                # END db22

                # Average density of nanoparticles
                with db23.container(border = True, height = heightCol - 75):                
                    x = np.arange(5, 100, 1)
                    averageDensity = NanoStat.averageDensityInNeighborhood(x, fullDist)

                    if (st.session_state['scale'] is not None):
                        fig = px.bar(x = x * st.session_state['scale'], y = averageDensity)
                    else:
                        fig = px.bar(x = x , y = averageDensity)


                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Average density of nanoparticles", font = dict(size=27)),
                        xaxis_title_text = 'Nanoparticle neighborhood size, nm',
                        yaxis_title_text = 'Average density of nanoparticles in neighborhood',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hoverinfo = "x+y",
                        hovertemplate = "Size: %{x:.2} nm <br>Density: %{y:.2}",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 0.5
                    )

                    st.plotly_chart(fig, use_container_width = True)
                # END db23
            
            with st.expander("Quality evaluation", icon = ":material/verified:"):
                instruct.AboutSectioQuality()
                
                if selectionUseNano == 1:
                    st.warning(tooltips.Warnings.NowUsingCVAT)

                uploadedGT = st.file_uploader("Expert markup file", type = ["csv", "zip"],
                    help = tooltips.ExpertFileUploader
                )
                            
                if uploadedGT is not None:
                    gt_blobs = None

                    if uploadedGT.type == 'text/csv':
                        string_data = io.StringIO(uploadedGT.getvalue().decode("utf-8"))
                        reader = csv.reader(string_data, delimiter = ',')
                        gt_blobs = np.array(list(reader), dtype=float) 

                        gt_blobs[:, 2] = gt_blobs[:, 2] * 2

                    elif (uploadedGT.type == 'application/zip') or (uploadedGT.type == 'application/x-zip-compressed'):                                    
                        gt_blobs, _, _ = API2CVAT.ImportTaskFromCVAT(uploadedGT) 
                    else:
                        raise ValueError("!")


                    if (gt_blobs is not None) and (st.session_state['statBLOBs'] is not None):
                        roi = accuracy.blobs2roi(gt_blobs, st.session_state['sizeImage'][1], st.session_state['sizeImage'][0])

                        accuracyPlaceholder = st.empty()

                        l, r = st.columns([2, 1])

                        with r:
                            if st.toggle("Duplicate filtering settings", disabled = True if selectionUseNano == 1 else False):
                                temp_brightness = st.slider("Nanoparticle center brightness", value = 10)

                                temp_diameter = st.slider("Nanoparticle diameter, px",
                                    value = (np.min(st.session_state['BLOBs_data'][:, 2]), np.max(st.session_state['BLOBs_data'][:, 2])),
                                    min_value = 0.5,
                                    step = 0.25,
                                    format = "%0.1f"
                                )

                                temp_reliability = st.slider("Nanoparticle reliability",
                                    value = 0.7,
                                    min_value = 0.0,
                                    step = 0.01,
                                    max_value = 1.0
                                )

                                params_filter_2 = {
                                    "thr_c0": temp_brightness,
                                    "min_thr_d": temp_diameter[0],   
                                    "max_thr_d": temp_diameter[1], 
                                    "thr_error": 1 - temp_reliability, 
                                }
            
                                # Filtering
                                temp_Filt_BLOBs_data = ExpApp.my_FilterBlobs_change(
                                    st.session_state['BLOBs_data'],
                                    params_filter_2
                                )


                                temp_filt_BLOBs = []
                                if (temp_Filt_BLOBs_data.shape[0] != 0):
                                    temp_filt_BLOBs = temp_Filt_BLOBs_data[:, :3]

                                # ОЧЕНЬ ПЛОХО!!! :P
                                st.session_state['statBLOBs'] = np.array(temp_filt_BLOBs)

                        temp_res = accuracy.accur_estimationDiametr(gt_blobs, st.session_state['statBLOBs'], roi, 0.25)                        
                        match, no_match, fake, FN, FP, TP, _ = temp_res

                        accuracyPlaceholder.markdown(f"""
                            <p class = 'text center'>
                                Accuracy: {match / (match + no_match + fake) * 100:.2f}%
                                (TP {match}; FN {no_match}; FP {fake})
                            </p>
                        """, unsafe_allow_html = True)

                        with l:
                            if st.toggle("Display nanoparticles"):
                            
                                fig = go.Figure()

                                fig.add_trace(go.Heatmap(
                                    z = np.array(st.session_state['statImage'].convert("L")),
                                    colorscale = 'gray',
                                    hoverinfo = 'skip',  
                                    showscale = False,   
                                ))
                            
                                ALL, _ = accuracy.blobs_in_roi(st.session_state['statBLOBs'], roi)

                                color_list = ['blue', 'green', 'red', 'yellow']
                                BLOBs_list = [ALL, TP, FN, FP]
                                shapes_list = [
                                    {
                                        'type': 'circle',
                                        'x0': x-d/2, 'y0': y-d/2, 'x1': x+d/2, 'y1': y+d/2,
                                        'line': {'width': 1.0, 'color': temp_color}
                                    }
                                    for temp_BLOBs, temp_color in zip(BLOBs_list, color_list)
                                    for y,x,d in zip(*temp_BLOBs.T)
                                ]
                                fig.update_layout(shapes = shapes_list, height = 600)


                                temp_gt_blobs = gt_blobs[:, 2] * st.session_state['scale']
                                fig.add_trace(go.Scatter(
                                    x = gt_blobs[:, 1],
                                    y = gt_blobs[:, 0],
                                    mode = 'markers',
                                    marker = dict(size = 15, opacity = 0),  
                                    hovertemplate = ("labeled <br>"
                                        "x: %{x:.1f} px<br>" +
                                        "y: %{y:.1f} px<br>" +
                                        "d: %{customdata[0]:.2f} px (%{customdata[1]:.2f} nm)<extra></extra>"
                                    ),
                                    customdata = list(zip(gt_blobs[:, 2], temp_gt_blobs)),
                                    showlegend = False
                                ))

                                temp_ALL = ALL[:, 2] * st.session_state['scale']
                                fig.add_trace(go.Scatter(
                                    x = ALL[:, 1],
                                    y = ALL[:, 0],
                                    mode = 'markers',
                                    marker = dict(size = 15, opacity = 0),  
                                    hovertemplate = ("detected <br>"
                                        "x: %{x:.1f} px<br>" +
                                        "y: %{y:.1f} px<br>" +
                                        "d: %{customdata[0]:.2f} px (%{customdata[1]:.2f} nm)<extra></extra>"
                                    ),
                                    customdata = list(zip(ALL[:, 2], temp_ALL)),
                                    showlegend = False
                                ))

                                # fig.add_shape(type = "rect",
                                #     xref = "x", yref = "y",
                                #     x0 = roi[1], y0 = roi[0],
                                #     x1 = roi[1] + roi[3], y1 = roi[0] + roi[2],
                                #     line = dict(
                                #         color = "red",
                                #         width = 4,
                                #         dash = "dot",
                                #     )
                                # )

                                fig.update_coloraxes(showscale = False)
                                fig.update_layout(
                                    margin = marginChart,
                                    hovermode = 'closest',
                                    xaxis_title = None,
                                    yaxis_title = None,
                                    xaxis = dict(showticklabels = False),
                                    yaxis = dict(showticklabels = False))
                                fig.update_xaxes(range = [roi[1], roi[1] + roi[3]], constrain='domain', scaleanchor = "y", scaleratio = 1)
                                fig.update_yaxes(range = [roi[0] + roi[2], roi[0]], constrain='domain')
      
                                instruct.LegendChartQuality()

                                st.plotly_chart(fig, use_container_width = True)
            
        with st.expander("Structured nanoparticles (experimental)", icon = ":material/pattern:"):  
            tempFile = st.file_uploader("File with coords particles")
            
            if tempFile is not None:
                currentSettings = structured.Parameters()

                with st.form('structured-parameters'):
                    l, c, r = st.columns([1, 1, 1], gap = 'large')
                    l2, cr2 = st.columns([1, 2], gap = 'large')

                    l.subheader("Parameters for prevailing directions", anchor = False)
                    currentSettings.DENSITY_NEIGHBOUR_COUNT = l.slider("DENSITY_NEIGHBOUR_COUNT",
                        min_value = 1, value = 3, max_value = 25)
                    currentSettings.DENSITY_WEIGHT = l.slider("DENSITY_WEIGHT",
                        min_value = 0.0, value = 1.5, max_value = 3.0)
                    currentSettings.PCA_NEIGHBOUR_COUNT = l.slider("PCA_NEIGHBOUR_COUNT",
                        min_value = 3, value = 8, max_value = 25)
                    currentSettings.THR_QUALITY = l2.slider("THR_QUALITY",
                        min_value = 0.0, value = 0.85, max_value = 1.0)
            
                    c.subheader("Parameters for lines construction by SUP", anchor = False)
                    currentSettings.MIN_LINE_SUP_LENGTH = c.slider("MIN_LINE_SUP_LENGTH",
                        min_value = 5, value = 10, max_value = 20)
                    currentSettings.WEIGHT_METRIC_THR = c.slider("WEIGHT_METRIC_THR",
                        min_value = 0.0, value = 0.03, max_value = 1.0)
                    currentSettings.WEIGHT_COAXIS = c.slider("WEIGHT_COAXIS",
                        min_value = 0.0, value = 1.5, max_value = 3.0)            
                    currentSettings.COAXIS_PERIOD = cr2.slider("COAXIS_PERIOD",
                        min_value = 1, value = 6, max_value = 12)

                    r.subheader("Parameters for lines construction by MSF", anchor = False)
                    currentSettings.MIN_LINE_MSF_LENGTH = r.slider("MIN_LINE_MSF_LENGTH",
                        min_value = 5, value = 10, max_value = 20)
                    currentSettings.MAX_DISTANCE = r.slider("MAX_DISTANCE",
                        min_value = 5.0, value = 20.0, max_value = 50.0)
                    currentSettings.NUMBER_LONGEST_LINE = r.slider("NUMBER_LONGEST_LINE",
                        min_value = 5, value = 20, max_value = 50)
                
                    st.form_submit_button("Apply and recalculate", disabled = tempFile is None)
            
                string_data = io.StringIO(tempFile.getvalue().decode("utf-8"))
                reader = csv.reader(string_data, delimiter = ';')
                next(reader); next(reader)
                BLOBs = np.array(list(reader), dtype=float)
                BLOBs[:, 2] = BLOBs[:, 2] / 2
                points2D = BLOBs[:, :2]



                if (st.session_state['struct'] is None):
                    st.session_state['struct'] = structured.Structured(points2D, currentSettings)
                else:
                    st.session_state['struct'].settings(currentSettings)

                l, c, r = st.columns([1, 1, 1], gap = 'large')
                
                f1, f2, f3 = st.session_state['struct'].featuresPrevailingDirections
                l.markdown(f"""
                    <div class = 'text' style = "text-align: center;">
                        Features prevailing directions: {f1:.2f}, {f2:.2f}, {f3:.2f} <br>
                    </div>
                """, unsafe_allow_html = True)
                
                f4, f5, f6, f7 = st.session_state['struct'].featuresLineSUP
                c.markdown(f"""
                    <div class = 'text' style = "text-align: center;">
                        Features SUP-lines: {f4:d}, {f5:.2f}, {f6:.2f} {f7:.2f}<br>
                    </div>
                """, unsafe_allow_html = True)

                mst = structured.kruskal_mst(points2D)
                forest = structured.mst_to_forest(mst, currentSettings.MAX_DISTANCE)
                forest_clean = structured.remove_terminal_to_highdegree_edges(forest, len(points2D))
                segments = structured.extract_segments(forest_clean, len(points2D))
                if (currentSettings.COAXIS_PERIOD > currentSettings.MIN_LINE_MSF_LENGTH):
                    currentSettings.COAXIS_PERIOD = currentSettings.MIN_LINE_MSF_LENGTH
                f8, f9 = structured.coaxis_all_segments_two_modes_threshold(points2D, segments, currentSettings.COAXIS_PERIOD, currentSettings.MIN_LINE_MSF_LENGTH)
                f10 = structured.count_points_segments_over_threshold(segments, currentSettings.MIN_LINE_MSF_LENGTH) / points2D.shape[0]
                f11 = structured.average_length_top_segments(segments, currentSettings.NUMBER_LONGEST_LINE)
                r.markdown(f"""
                    <div class = 'text' style = "text-align: center;">
                        Features MSF-lines: {np.mean(f8):.2f}, {np.mean(f9):.2f}, {f10:.2f} {f11:.2f}<br>
                    </div>
                """, unsafe_allow_html = True)


                # display prevailing directions
                fig, ax = plt.subplots(1, 1, sharex = True, sharey = True)
                structured.showBlobs(BLOBs, ax)

                k, quality = st.session_state['struct'].prevailingDirections

                for i, tempPoint in enumerate(BLOBs):
                    x = tempPoint[1]
                    y = tempPoint[0]
    
                    dx = 6 / np.sqrt(k[i]**2 + 1)
                    dy = 6 / np.sqrt(1 + 1/k[i]**2)

                    if k[i] > 0:
                        plt.plot([x+dy, x-dy], [y-dx, y+dx], 'r', alpha = quality[i], lw=1.0)
                    else:            
                        plt.plot([x-dy, x+dy], [y-dx, y+dx], 'r', alpha = quality[i], lw=1.0)
                
                plt.gca().invert_yaxis()
                l.pyplot(fig, clear_figure = True)

                # display lines construction by SUP
                fig, ax = plt.subplots(1, 1, sharex = True, sharey = True)
                structured.showBlobs(BLOBs, ax)

                for i, tempLine in enumerate(st.session_state['struct'].lineSUP):
                    y, x, _ = BLOBs[tempLine.start.index]
                    ax.add_patch(plt.Circle((x, y), 3, color = 'g', linewidth = 1.5, fill = True)) 

                    plt.text(x+2, y-2, str(i), color = 'k', fontsize = 6)
                    plt.plot(tempLine[:, 1], tempLine[:, 0], color = 'k')
                    
                plt.gca().invert_yaxis()
                c.pyplot(fig, clear_figure = True)

                # display lines construction by MSF
                fig, ax = plt.subplots()
                structured.showBlobs(BLOBs, ax)
                structured.visualize_forest_with_long_segments(ax,
                    BLOBs,
                    forest_clean,
                    segments,
                    min_length = currentSettings.MIN_LINE_MSF_LENGTH
                )
                r.pyplot(fig, clear_figure = True)
                
                tempModel = load_SVM_model()
                
                tempFeatures = [[np.mean(f8), np.mean(f9), f10, f11, f2, f3, f7]]
                Y_pred = tempModel.predict(tempFeatures)
                Y_prob = 1 / (1 + np.power(1.5, -tempModel.decision_function(tempFeatures)))
                
                tempClass = 'ordered (defects)' if Y_pred > 0 else 'disordered (not defects)'
                tempProb = (Y_prob[0] if Y_pred > 0 else (1 - Y_prob[0])) * 100

                st.markdown(f"""
                    <div class = 'text' style = "text-align: center;">
                        Estimated сlass: <b>{tempClass}</b>.
                        Probability of belonging to <b>{tempClass}</b> class: {tempProb:.2f}%
                    </div>
                """, unsafe_allow_html = True)


    ## TAB 3
    with tabHelp:
        if st.button("If you have any difficulties with our tool, please contact us (click here)",
            key = 'button_contact',
            use_container_width = True
        ):            
            st.warning(tooltips.Warnings.ReportLimit)
            #dialog_feedback()


        # Guide 1: Detection and filtration of nanoparticles
        instruct.Guide1()
                
        # Guide 2: Interaction with detection results
        instruct.Guide2()

        # Guide 3: Integration with CVAT
        instruct.Guide3()        

        # Guide 4: Evaluation of detection quality
        instruct.Guide4()        
        
    
    ## How to cite
    instruct.HowCite()
    
    ## Footer
    instruct.Footer()

except Exception as exc:
    dialog_exception(False) # passing email with error 