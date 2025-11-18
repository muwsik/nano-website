# Run application
# streamlit run .\nano-website\nano-website.py

import streamlit as st

import io, csv
import cv2, skimage, scipy
import numpy as np
import time, datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

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


import traceback
    

### Function ###
    
help_str = "hint be added soon"

colorRGBA_str = 'rgb(150, 150, 255)'
colorRGB = (75, 255, 75)


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
    style.set_style(colorRGBA_str)
    
    # Initial loading of session states
    if 'rerun' not in st.session_state:
        loadDefault_sessionState()
    elif st.session_state['rerun']:
        loadDefault_sessionState(True)
    
    ## Header
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
    
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
            <br>It will help you to detect nanoparticles in the image and calculate their statictics.
        </div>
    """, unsafe_allow_html = True)

    st.markdown("""
        <div style = "padding-bottom: 25px" class = 'about'>
            Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
        </div>
    """, unsafe_allow_html = True)


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
                st.toggle("Use default settings",
                    disabled = not st.session_state['imgUpload'],
                    key = 'settingDefault'
                )

                # Preprocessing image
                with st.spinner("Preprocessing image...", show_time = True):                           
                    
                    tempArrImg = np.array(srcImage, dtype = 'uint8')

                    st.session_state['scale'], st.session_state['scaleData'] = autoscale.estimateScale(
                        tempArrImg
                    )

                    lowerBound = autoscale.findBorder(tempArrImg)                    
                    if (lowerBound is not None):
                        srcImage = srcImage.crop((0, 0, srcImage.size[0], lowerBound))
                        
                    st.session_state['sizeImage'] = srcImage.size

                    if (st.session_state['typeImg'] is None):
                        data = np.array(srcImage, dtype = 'uint8').flatten()
                        counts, _ = np.histogram(data, bins = np.arange(0, 255, 1))
                        counts = counts / np.sum(counts)

                        cumSum = 0
                        for i, j in enumerate(counts):
                            cumSum = cumSum + j
                            if (cumSum >= 0.5):
                                if (i <= 127):
                                    st.session_state['typeImg'] = 'SEM'
                                else: st.session_state['typeImg'] = 'TEM'
                                break
                            
                    if (st.session_state['typeImg'] == 'TEM'):
                        srcImage = ImageOps.invert(srcImage)

                            
                # Detection settings       
                with st.expander("Detection settings", expanded = not st.session_state['detected'], icon = ":material/tune:"):
                    if ('param-pre-1' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-1'] = 10
                    
                    st.slider("Nanoparticle brightness",
                        key = 'param-pre-1',
                        disabled = st.session_state['settingDefault'],
                        help = "The average brightness of nanoparticles and its surroundings in the image"
                    )

                    option_nanoparticleSize = {
                        0: "Small (1-10 pixels)",
                        1: "Medium (10-20 pixels)",
                        2: "Large (20-35 pixels)"      
                    }

                    if ('param-pre-2' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-2'] = 1

                    st.selectbox("Hypothetical nanoparticles diameter",
                        key = 'param-pre-2',
                        index = 0,
                        options = option_nanoparticleSize.keys(),
                        format_func = lambda option: option_nanoparticleSize[option],
                        disabled = st.session_state['settingDefault'],
                        help = "Hipotetical diameter of nanoparticles in pixels"
                    )

                    if ('param-pre-3' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-pre-3'] = True

                    st.toggle("Suppression of background irregularities",
                        key = 'param-pre-3',                              
                        disabled = st.session_state['settingDefault'],
                        help = help_str
                    )
        
                    pushDetectButton = st.button("Nanoparticles detection",
                        use_container_width = True,
                        disabled = not st.session_state['imgUpload'],
                        help = help_str,
                        on_click = update_sessionState,
                        args = ("detected", True)
                    )
                    
                    warningPlaceholder = st.empty()
                    if (st.session_state['detectionSettings'] is not None) and st.session_state['detected']:
                        if (st.session_state['detectionSettings'] != [st.session_state['param-pre-1'], st.session_state['param-pre-2'], st.session_state['param-pre-3']]):
                            warningPlaceholder.warning("""
                                The detection settings have been changed.
                                To accept the new settings, click the button "Nanoparticles detection"! 
                            """, icon = ":material/warning:")
                
                # Detecting
                if pushDetectButton:
                    st.session_state['detected'] = False
                    warningPlaceholder.empty()
                    
                    timeStart = time.time()
                    with st.spinner("Nanoparticles detection...", show_time = True):                    
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
                                "sigma_gauss": 0.5,
                                # размер диска Top-Hat
                                "sz_th":  7,
                                # порог яркости для отбрасывания лок. максимумов
                                "thr_br": float(st.session_state['param-pre-1']),   
                                # минимальное расстояние между локальными максимумами при их поиске 
                                "min_dist": 6,
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
                                "min_dist": 5,
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
                        st.session_state['detectionSettings'] = [st.session_state['param-pre-1'], st.session_state['param-pre-2'], st.session_state['param-pre-3']]
        
                    st.session_state['timeDetection'] = int(np.ceil(time.time() - timeStart))

                # Detection results
                if st.session_state['detected']:
                    temp_time = st.session_state['timeDetection']
                    st.markdown(f"""
                        <p class = 'text'>
                            Nanoparticles detected: <b>{st.session_state['detectedParticles']}</b> ({temp_time//60}m : {temp_time%60:02}s)
                        </p>""", unsafe_allow_html = True
                    )

                    # Warning about not correctly detection results 
                    if (st.session_state['detectedParticles'] < 1):            
                        st.warning("""
                            Nanoparticles not found!
                            Please change the detection settings or upload another SEM image!
                        """, icon = ":material/warning:")
                                    
                # Action with correctly detection results
                if (st.session_state['detected'] and st.session_state['detectedParticles'] > 0):
                    # Filtration settings
                    with st.expander("Filtration settings", expanded = True, icon = ":material/filter_alt:"):
                        if ('param-filt-1' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-1'] = 10
                                                    
                        st.slider("Nanoparticle center brightness",
                            key = 'param-filt-1',
                            disabled = st.session_state['settingDefault'],
                            help = "Brightness in the central pixel of the nanoparticle"
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
                            disabled = st.session_state['settingDefault']
                        )

                        if ('param-filt-3' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-3'] = 0.65

                        st.slider("Nanoparticle reliability",
                            key = 'param-filt-3',
                            min_value = 0.0,
                            step = 0.01,
                            max_value = 1.0,
                            disabled = st.session_state['settingDefault'],
                            help = "The higher the reliability, the clearer the nanoparticle is against the background of the image"
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
                                help = help_str
                            )

                            temp_img = ExpApp.PreprocessingMedian(st.session_state['srcImg'].copy(), 3)
                            temp_img = ExpApp.PreprocessingTopHat(temp_img, 7)   

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
                        st.warning("""
                            There are no nanoparticles satisfying the filtration settings!
                            Please change the filtering settings!
                        """, icon = ":material/warning:")
                                      
                    # Info about filtered nanoparticles
                    st.markdown(f"""
                        <p class = 'text'>
                            Nanoparticles after filtration: <b>{st.session_state['filteredParticles']}</b>
                        </p>""", unsafe_allow_html = True
                    )
                                        
                    with st.expander("Visualization and saving results", expanded = False, icon = ":material/display_settings:"):
                        # Displaying the scale
                        st.toggle("Estimated scale display", key = 'displayScale', disabled = False, help = help_str)

                        if (st.session_state['displayScale'] and st.session_state['scaleData'] is None):
                            st.warning("""
                                The image scale could not be determined automatically!
                                Using default scale: 1.0 nm/px
                            """, icon = ":material/warning:")                    

                        # Slider for comparing the results before and after detection
                        st.toggle("Comparison mode", value = True, key = 'comparison', disabled = False, help = help_str)

                        #
                        st.toggle("Display area of background irregularities", key = 'areas', disabled = False,
                            help = "The areas with background irregularities are colored red"
                        )
                            
                        # Saving
                        selectboxCol, buttonCol = st.columns([6,1], vertical_alignment = 'bottom')

                        option_saving = {
                            0: "Particles on clear background (*.tif)",
                            1: "Particles on EM-image (*.tif)",
                            2: "Particles characteristics (*.csv)",
                            3: "CVAT task (*.zip)"
                        }

                        selectionSave = selectboxCol.selectbox(
                            "What results should be saved?",
                            index = 2,
                            placeholder = "Select options...",
                            options = option_saving.keys(),
                            format_func = lambda option: option_saving[option]
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
                            disabled = button_download_disabled
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
              
        with st.expander("Global dashboard settings", expanded = not st.session_state['calcStatictic'], icon = ":material/rule_settings:"):
            option_map = {
                0: "Automatically detected",
                1: "Import from CVAT"
            }

            selection_use = st.selectbox(
                "Which nanoparticles to use?",
                index = 1,
                options = option_map.keys(),
                format_func = lambda option: option_map[option]
            ) 
                
            match selection_use:
                case 0:
                    if (not st.session_state['detected']):
                        st.session_state['calcStatictic'] = False
                        st.warning("""
                            Nanoparticle detection is necessary to calculate their statistics.
                            Please go to "Automatic detection" tab. """, icon = ":material/warning:")
                    elif (st.session_state['filteredParticles'] < 10):
                        st.session_state['calcStatictic'] = False
                        st.warning("""
                            Nanoparticles after detection and filtration are less than 10! 
                            Please go to the "Detection" tab and change the detection,
                            filtering settings or upload another SEM image! """, icon = ":material/warning:")
                    else:                        
                        st.session_state['calcStatictic'] = True
                        st.session_state['statBLOBs'] = st.session_state['BLOBs_filter']
                        st.session_state['statImageName'] = Path(uploadedImg.name).stem
                        st.session_state['statImage'] = st.session_state['srcImg'].convert('RGB')
                case 1:
                    st.markdown(f"""
                        Import <a href='https://app.cvat.ai/'>CVAT</a> data to calculate statistics (format 'CVAT for images 1.1')
                        """, unsafe_allow_html = True)

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
                        st.session_state['scale'], st.session_state['scaleData'] = autoscale.estimateScale(st.session_state['statImage'].convert("L"))
                        
                        if st.session_state['scale'] is not None:
                            st.session_state['sizeImage'] = (st.session_state['statImage'].size[0], st.session_state['scaleData'][0])
                        else:
                            st.session_state['sizeImage'] = st.session_state['statImage'].size
                case _:
                    pass

        if (not st.session_state['calcStatictic']):
            defaultStatTab()
        else:
            with st.expander("Particle parameters", expanded = True, icon = ":material/app_registration:"):
                st.markdown(f"""
                    <p class = 'text'>
                        The main parameters of nanoparticles can be represented as primary values: 
                        the average diameters, its deviations, or a histogram of the diameters distribution. 
                        Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
                        which can be normalized to the area of the SEM image.
                    </p>""", unsafe_allow_html = True
                )
                                
                boolIndexSelectedBLOBs = None   

                diameter_nm = st.session_state['statBLOBs'][:, 2]  
                if st.session_state['scale'] is not None:
                    diameter_nm = diameter_nm * st.session_state['scale']

                db11, db12, db13 = st.columns([4, 4, 4])            

                # Particle size distribution
                with db11.container(border = True, height = heightCol):
                    left, rigth = st.columns([7, 1])
                    left.subheader("Distribution of particle diameters", anchor = False)

                    with rigth.popover("", icon=":material/settings:"):
                        st.toggle("Display distribution function",
                            key = 'distView',
                            help = help_str
                        )

                        st.toggle("Normalize the vertical axis",
                            key = 'normalize',
                            help = help_str
                        )

                        st.toggle("Selecting individual columns",
                            key = 'selection',
                            help = help_str
                        )

                        st.number_input("Histogram step",
                            key = 'step',
                            min_value = 0.1,
                            max_value = 1.0,
                            step = 0.1,
                            format = '%0.2f',
                            value = 0.5
                        )
                        
                        buttonPlaceholder = st.empty()
                          
                    
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

                    
                    buttonPlaceholder.download_button(
                        label = "Download data chart *.csv",
                        data = file.getvalue(),
                        file_name = st.session_state['statImageName'] + "-dist-diameters.csv",
                        use_container_width  = True,
                        help = help_str
                    )

                    fig = go.Figure()

                    fig = fig.add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = bar_y,
                        customdata = customDataChart,
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
                            name = '',
                            hoverinfo = 'skip',
                            line = dict(color = 'rgba(0,0,255,0.75)')
                        ))         

                    fig.update_layout(
                        margin = marginChartLess,
                        xaxis_title_text = name_x,
                        yaxis_title_text = name_y,
                        showlegend = False,
                        bargap = 0
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

                # Nanoparticle parameters
                with db12.container(border = True, height = heightCol):                        
                    
                    left, rigth = st.columns([7, 1])
                    left.subheader("Nanoparticle parameters", anchor = False)

                    with rigth.popover("", icon=":material/settings:"):
                        option_materialDensity = {
                            0: "Palladium (Pd)",    # 12.02 * 10**-12 ng / nm^3
                            1: "Cuprum (Cu)",       #  8.96 * 10**-12 ng / nm^3
                            2: "Alloy 30% Au + 70% Pd (AuPd)",  # 14.10 * 10**-12 ng / nm^3
                            3: "Alloy 70% Cu + 30% Zn (CuZn)",  #  8.42 * 10**-12 ng / nm^3
                            4: "User density"
                        }

                        selectionDensity = st.selectbox(
                            "Particles material",
                            index = 0,
                            placeholder = "Select material...",
                            options = option_materialDensity.keys(),
                            format_func = lambda option: option_materialDensity[option],
                        )

                        match selectionDensity:
                            case 0: materialDensity = 12.02 * 10**-12
                            case 1: materialDensity = 8.96 * 10**-12
                            case 2: materialDensity = 14.10 * 10**-12
                            case 3: materialDensity = 8.42 * 10**-12
                            case 4: pass;
                            case _: materialDensity = 1
                        
                        if selectionDensity == 4:
                            materialDensity = st.number_input(
                                "Particles material density on ng/nm³",
                                min_value = 0.0,
                                step = 1.0e-11,
                                value = 1.0e-10,
                                format = "%0.2e",
                                key = "user-density"
                            )
                        else:
                            st.markdown(f"""
                                <div class = 'text' style = "font-size: 16px;">
                                    Particles material density: <b>{materialDensity:.2e} ng/nm<sup>3</sup></b> 
                                </div>""", unsafe_allow_html = True)
                      
                    if st.session_state['scale'] is None:
                        pass
                    else:                    
                        st.markdown(f"""
                            <div class = 'text'>
                                Estimated scale: <b>{st.session_state['scale']:.3f} nm/px</b> 
                            </div>""", unsafe_allow_html = True)
                    
                    if selectionDensity != 4:
                        st.markdown(f"""
                            <div class = 'text'>
                                Material: <b>{option_materialDensity[selectionDensity]}</b> 
                            </div>""", unsafe_allow_html = True)
                    else:
                        st.markdown(f"""
                            <div class = 'text'>
                                Material: <b>{option_materialDensity[selectionDensity]} ({materialDensity:.2e} ng/nm<sup>3</sup>)</b> 
                            </div>""", unsafe_allow_html = True)



                    temp_add_str = ""
                    currentDiameter = diameter_nm
                    if boolIndexSelectedBLOBs is not None:
                        currentDiameter = currentDiameter[boolIndexSelectedBLOBs]
                        temp_add_str = f"(includ {len(currentDiameter)} selected)"
   
                    st.markdown(f"""
                        <div class = 'text'>
                            Quantity: <b>{len(st.session_state['statBLOBs'])}</b>
                            {temp_add_str}
                        </div>""", unsafe_allow_html = True)

                    st.subheader("Primary parameters", anchor = False)                    
                    st.markdown(f"""
                        <div class = 'text'>
                            Average diameter: <b>{np.mean(currentDiameter):.3f} nm</b> 
                        </div>""", unsafe_allow_html = True)
                    st.markdown(f"""
                        <div class = 'text'>
                            Standart deviation diameters: <b>{np.std(currentDiameter):.3f} nm</b> 
                        </div>""", unsafe_allow_html = True)


                    st.subheader("Secondary parameters", anchor = False)  
                    volumeParticls = (np.pi * currentDiameter**3) / 6
                    areaParticls =  np.sum((np.pi * currentDiameter**2) / 4)
                    massParticls = np.sum(volumeParticls * materialDensity)

                    st.markdown(f"""
                        <div class = 'text'>
                            Mass: <b>{massParticls:0.2e} ng</b> 
                        </div>""", unsafe_allow_html = True)   
                    
                    st.markdown(f"""
                        <div class = 'text'>
                            Volume: <b>{np.sum(volumeParticls):0.2e} nm<sup>3</sup></b> 
                        </div>""", unsafe_allow_html = True) 
                    
                    st.markdown(f"""
                        <div class = 'text'>
                            Area: <b>{areaParticls:0.2e} nm<sup>2</sup></b> 
                        </div>""", unsafe_allow_html = True)
                    

                    imageArea = np.prod(st.session_state['sizeImage'])
                    if st.session_state['scale'] is not None:
                        imageArea = imageArea * st.session_state['scale']**2

                    st.subheader("Secondary parameters (norm)",
                        help = f"Values relative to the surface area is {imageArea:.2e} nm²",
                        anchor = False
                    )                    
                       
                    st.markdown(f"""
                        <div class = 'text'>
                            Area: <b>{areaParticls/imageArea*100:0.2f}</b> %
                            </div>""", unsafe_allow_html = True)

                    st.markdown(f"""
                        <div class = 'text'>
                            Mass: <b>{massParticls/imageArea:0.2e} ng/nm<sup>2</sup></b> 
                        </div>""", unsafe_allow_html = True)                        
                # END db12

                # Heatmap of particle count
                # or
                # Visualization particles
                with db13.container(border = True, height = heightCol):
                    option_typeChart = {
                        0: "Heatmap of particle count",
                        1: "Visualization particles",
                    }

                    selectionUse = st.selectbox(
                        "Type chart",
                        index = 1,
                        options = option_typeChart.keys(),
                        format_func = lambda option: option_typeChart[option],
                        label_visibility = 'collapsed'
                    )

                    match selectionUse:
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
                st.markdown(f"""
                    <p class = 'text'>
                        Visual representation of nanoparticle-based statistics in image.
                        A detailed description is provided in the work on the [2] link below.
                    </p>""", unsafe_allow_html = True)

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
            
            with st.expander("Quality evaluation", expanded = False, icon = ":material/verified:"):
                st.markdown(f"""
                    <p class = 'text'>
                        Quality evaluation of the automatically detected nanoparticles 
                        based on the Jacquard measure and the expert's manual marking.
                        A detailed description is provided in the work on the [2] link below.
                    </p>""", unsafe_allow_html = True)
                
                if selection_use == 1:
                    st.warning("""This section is designed for evaluating automated nanoparticle detection algorithms. 
                        Currently using data imported from CVAT - please verify data accuracy before proceeding.""")

                uploadedGT = st.file_uploader("Expert markup file", type = ["csv", "zip"],
                    help = f"""If file is *.CSV, then each line format 'y, x, r' is a nanoparticle.
                    If file is *.ZIP, it must match the form CVAT for image 1.1."""
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

                        temp_res = accuracy.accur_estimationDiametr(gt_blobs, st.session_state['statBLOBs'], roi, 0.25)                        
                        match, no_match, fake, FN, FP, TP, _ = temp_res

                        st.write(f"""
                            Accuracy: {match / (match + no_match + fake) * 100:.2f}%
                            (TP {match}; FN {no_match}; FP {fake})""")


                        if st.toggle("Display nanoparticles"):
                            
                            fig = go.Figure()

                            fig.add_trace(go.Heatmap(
                                z = np.array(st.session_state['statImage'].convert("L")),
                                colorscale = 'gray',
                                hoverinfo = 'skip',  
                                showscale = False,   
                            ))
                            
                            ALL = accuracy.blobs_in_roi(st.session_state['statBLOBs'], roi)[0]

                            color_list = ['blue', 'green', 'red', 'yellow']
                            BLOBs_list = [ALL, TP, FN, FP]
                            shapes_list = [
                                {
                                    'type': 'circle',
                                    'x0': x-d/2, 'y0': y-d/2, 'x1': x+d/2, 'y1': y+d/2,
                                    'line': {'width': 0.75, 'color': temp_color}
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


                            fig.add_shape(type = "rect",
                                xref = "x", yref = "y",
                                x0 = roi[1], y0 = roi[0],
                                x1 = roi[1] + roi[3], y1 = roi[0] + roi[2],
                                line = dict(
                                    color = "red",
                                    width = 4,
                                    dash = "dot",
                                )
                            )

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
      
                            st.markdown("""
                                <div style = "text-align: center;">
                                    By algorithm particles is:<br>
                                    <span style = "color: white; background-color: #007bff; padding: 2px 6px; border-radius: 4px; font-weight: bold;">All detected</span>
                                    <span style = "color: white; background-color: #28a745; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Correctly identified (TP)</span>
                                    <span style = "color: white; background-color: #dc3545; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Not identified (FN)</span>
                                    <span style = "color: white; background-color: #fd7e14; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Identified but not confirmed by expert (FP)</span>
                                </div>
                            """, unsafe_allow_html=True)

                            st.plotly_chart(fig, use_container_width = True)


    ## TAB 3
    with tabHelp:
        if st.button("If you have any difficulties with our tool, please contact us (click here)",
            key = 'button_contact',
            use_container_width = True
        ):
            dialog_feedback()


        # Guide 1
        st.subheader("Детектирование и фильтрация наночастиц", anchor = False)
        text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

        text_col.markdown(f"""
            <div>
                <p class = 'text'>Все дальнейшие шаги выполняются на вкладке «Automatic detection».</p>
                <ul>
                    <li>
                        <p class = 'text'>
                            Шаг 1. Загрузка исходного СЭМ-изображения (кнопка «Browse file»).
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Шаг 2. Детектирование наночастиц (кнопка «Nanoparticles detection» становится активной после
                            загрузки изображения). Процесс детектирования занимает некоторое время, в среднем до одной минуты.                     
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Шаг 3. После успешного детектирования производится фильтрация найденных наночастиц
                            (используются параметры по умолчанию). Отфильтрованные частицы отображаются
                            на изображении в виде окружностей.
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Шаг 4. Можно вручную изменять параметры детектирования и фильтрации наночастиц
                            (снять галочку «Use default settings»). <strong>ВАЖНО:</strong> подтверждение параметров 
                            детектирования осуществляется повторным нажатием кнопки «Nanoparticles detection». Параметры
                            фильтрации применяются автоматически.
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)

        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)
                
        # Guide 2
        st.subheader("Взаимодейтсвие с результатами детектирования", anchor = False)
        text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

        text_col.markdown(f"""
            <div>
                <p class = 'text'>Указанный функционал доступен на вкладке «Automatic detection» после детектирования наночастиц.</p>
                <ul>
                    <li>
                        <p class = 'text'>
                            Результаты детектирования можно скачать в нескольких вариантах:
                            (1) Найденные частицы на прозрачном фоне. (2) Найденные частицы, наложенные на исходное изображение.
                            (3) Файл с указанием координат центра и радиуса каждой частицы.
                            Для этого нужно в выпадающем списке «What results should be saved?» выбрать нужный вариант 
                            и нажать кнопку, расположенную правее.
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Если на изображении присутствует мерная шкала и указан её физический размер, масштаб 
                            определяется автоматически. Визуализировать вычисленный масштаб можно с помощью 
                            переключателя «Display scale».
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Режим сравнения в доработке!
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)
       
        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)

        # Guide 3
        st.subheader("Интеграция с CVAT", anchor = False)
        text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

        text_col.markdown(f"""
            <div>              
                <ul>
                    <li>
                        <p class = 'text'>
                            Результаты детектирования можно скачать в формате, поддерживаемом <a href=https://app.cvat.ai/>CVAT</a>.
                            Для этого на вкладке «Automatic detection» после детектирования наночастиц
                            нужно в выпадающем списке «What results should be saved?» выбрать
                            пункт «CVAT task» и нажать кнопку, расположенную правее. Скачанный backup-архив можно 
                            использовать для создания новой задачи CVAT.
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Разметку, полученную в CVAT, можно импортировать на сайт. Для этого сначала необходимо выгрузить
                            из CVAT backup-архив задачи с нужной разметкой. Затем на вкладке «Statistics dashboard» 
                            в выпадающем списке «Which nanoparticles to use» нужно выбрать пункт «Import from CVAT» и 
                            загрузить backup-архив в соответствующее поле. Если все условия выполнены, ниже автоматически 
                            отобразятся все разделы со статистикой.
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Более подробная информация об интеграции с CVAT приведена в 
                            <a href = "https://disk.yandex.ru/i/2U5wgJ8IjskREQ"
                                >расширенном руководстве</a>.
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)
       
        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)
        

        # Guide 4
        st.subheader("Оценка качества детектирования", anchor = False)
        text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

        text_col.markdown(f"""
            <div>
                <p class = 'text'>Все дальнейшие шаги выполняются на владке «Statistics dashboard».</p>
                <ul>                    
                    <li>
                        <p class = 'text'>
                            В разделе «Quality evaluation» можно получить численную оценку качества детектирования наночастиц.
                            Для этого, в первую очередь, необходим результат автоматического детектирования. Он должен быть
                            либо на вкладке «Automatic detection», либо в виде backup-архива CVAT, который нужно загрузить 
                            в разделе «Global dashboard settings». Далее требуется загрузить файл с экспертной разметкой, 
                            также в формате backup-архива CVAT, в соответствующее поле раздела «Quality evaluation». Если 
                            все условия выполнены, ниже отобразится качество в процентах. Подробно процедура оценки качества 
                            описана в работе [2].
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Дополнительно можно визуализировать результат оценки качества детектирования. Для этого
                            переключите тумблер «Display nanoparticles». В результате ниже появится интерактивный график,
                            на котором будут отмечены наночастицы четырёх типов: "Синие" - это автоматически детектированные
                            частицы, которые сопоставлены с "зелёными" наночастицами, отмеченными экспертом (TP).
                            "Красные" — это наночастицы, которые были помечены экспертом, но не были детектированы 
                            автоматически (FN). "Жёлтые" - это автоматически детектированные наночастицы, которые не были 
                            подтверждены экспертом (FP).
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)
       
        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)
    
    ## How to cite
    tempCol = st.columns([0.8, 0.2], vertical_alignment = 'center')

    tempCol[0].markdown("""
        <div class = 'cite'> <b>How to cite</b>:
            <ul>
                <li> <p class = 'cite'>
                    [1] An article about this site will be published soon, don't miss it!
                </p> </li>
                <li> <p class = 'cite'>
                    [2] Automated Recognition of Nanoparticles in Electron Microscopy Images of Nanoscale Palladium Catalysts.
                    Boiko D.A., Sulimova V.V., Kurbakov M.Yu. [et al.] 
                    // Nanomaterials. 2022. Vol. 12, No. 21. Pp. 3914. 
                    DOI: <a href=https://www.mdpi.com/2079-4991/12/21/3914>10.3390/nano12213914</a>.
                </p> </li>
                <li> <p class = 'cite'>
                    [3] Determining the Orderliness of Carbon Materials with Nanoparticle Imaging and Explainable Machine Learning. 
                    Kurbakov M.Yu., Sulimova V.V., Kopylov A.V. [et al.]
                    // Nanoscale. 2024. Vol. 16, No. 28. Pp. 13663-13676. 
                    DOI: <a href=https://pubs.rsc.org/en/content/articlelanding/2024/nr/d4nr00952e>10.1039/d4nr00952e</a>.
                </p> </li>                
                <li> <p class = 'cite'>
                    [4] Interpretable Graph Methods for Determining Nanoparticles Ordering in Electron Microscopy Images.
                    Kurbakov M.Yu., Sulimova V.V., Seredin O.S., Kopylov A.V. // Computer Optics. 2025. Vol. 49, No 3. Pp. 470-479.
                    DOI: <a href=https://computeroptics.ru/eng/KO/Annot/KO49-3/490313e.html>10.18287/2412-6179-CO-1568</a>.
                </p> </li>
            </ul>
        </div>""", unsafe_allow_html = True)
    tempCol[1].image(r"./nano-website/content/qr-code.svg",
        caption = "Web Nanoparticles QR-code",
        use_container_width = True
    )   
    
    ## Footer
    st.markdown(f"""
        <div class = 'footer'>
            Laboratory of Cognitive Technologies and Simulating Systems,
            Tula State University © {datetime.datetime.now().year} (E-mail: muwsik@mail.ru)
        </div>""", unsafe_allow_html = True)

except Exception as exc:
    dialog_exception()