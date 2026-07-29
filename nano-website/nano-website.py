# Run application
# streamlit run .\nano-website\nano-website.py

import streamlit as st

import io, csv
import cv2, skimage, scipy
import numpy as np
import pandas as pd
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

import plotly.express as px
import plotly.graph_objects as go
#import plotly.figure_factory as ff

#import matplotlib.pyplot as plt

# content
import content.style as style
import content.tooltips as tooltips
import content.instructions as instruct

# utils
import utils.autoscale as autoscale
import utils.NanoStatistics as NanoStat
import utils.ExponentialApproximation as ExpApp
import utils.ExponentialApproximation2 as ExpApp2
import utils.WebsiteBot as webBot
import utils.API2CVAT as API2CVAT
import utils.accuracy as accuracy

# dev utils
import utils.reworkExpApp as rEA
from streamlit_image_overlay import streamlit_image_overlay

import traceback
    

### Function ###
    
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
            
    st.session_state['detectParticles'] = None
    st.session_state['filterParticles'] = None

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
    st.session_state['scaleInfo'] = None

    defaultDetectTab()
    defaultStatTab()

    st.session_state["analysisBuffer"] = pd.DataFrame(columns = [
        "Image",
        "Number of particles",
        "Material type",
        "Mean particle diameter, nm",
        "Particle surface density, mg/m²",
        "Mean distance to neighbour, nm",
        "Most probable distance to neighbour, nm",
        "Distance threshold, nm",
        "Fraction below distance threshold",
        "Clark-Evans index (R)",
    ])


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

    
@st.cache_data(show_spinner = False, max_entries = 5)
def analyzeScaleRegion(p_image):
    return autoscale.analyzeScaleRegion(p_image)



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
        uploadedImg = st.file_uploader("Choose an SEM image",
            label_visibility = 'collapsed',
            type = ["tif", "tiff", "png", "jpg", "jpeg" ]
        )
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

            # Detection settings and results
            with colSetting: 
                # Preprocessing image
                with st.spinner("Preprocessing image...", show_time = True):                    
                    # defining the scale and related information
                    scale, lowerBound, st.session_state['scaleInfo'] = analyzeScaleRegion(srcImage)
                    st.session_state['scale'] = autoscale.Scale(scale)
                  
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
                                              
                st.toggle("Use default settings",
                    disabled = not st.session_state['imgUpload'],
                    key = 'settingDefault',
                    help = tooltips.DefaultToggle
                )                         

                # Detection settings       
                with st.expander("Detection settings", expanded = not st.session_state['detected'], icon = ":material/tune:"):
                    st.segmented_control("Type of microscope image",
                        required = True,
                        key = 'typeImg',
                        options = tooltips.Options.TypeMicroscope.keys(),
                        format_func = lambda option: tooltips.Options.TypeMicroscope[option],  
                        width = 'stretch',                     
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

                    
                    ### [Deprecated]
                    # if ('param-pre-3' not in st.session_state) or st.session_state['settingDefault']:
                    #         st.session_state['param-pre-3'] = False

                    # st.toggle("Suppression of background irregularities",
                    #     key = 'param-pre-3',                              
                    #     disabled = True, #st.session_state['settingDefault'],                        
                    #     help = tooltips.Detection.Irregularities
                    # )
        
                    pushDetectButton = st.button("Nanoparticles detection",
                        width = 'stretch',
                        disabled = not st.session_state['imgUpload'],
                        on_click = update_sessionState,
                        args = ("detected", True)
                    )
                    
                    tempWarningPlaceholder = st.empty()
                    tempSettings = [                        
                        st.session_state['typeImg'],
                        st.session_state['param-pre-1'],
                        st.session_state['param-pre-2'],
                        #st.session_state['param-pre-3']
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
                                #"deleteBorderLines": st.session_state['param-pre-3'], 
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
                                #"deleteBorderLines": st.session_state['param-pre-3'], 
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
                                #"deleteBorderLines": st.session_state['param-pre-3'], 
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

                        detectionParticles = rEA.blobs2Particles(blobs_appr)
                        
                        if (st.session_state['param-pre-2'] == 0) or (st.session_state['param-pre-2'] == 1):
                            pass
                        elif st.session_state['param-pre-2'] == 2:                            
                            blobs_appr[:, :3] = blobs_appr[:, :3] * 2

                            for _temp_ in detectionParticles:
                                _temp_.x *= 2
                                _temp_.y *= 2
                                _temp_.diameter *= 2
                        else:
                            raise ValueError("!")

                        blobs_appr[:, 2] = blobs_appr[:, 2] * 2         # radius -> diametr
                        blobs_appr[:, [5, 6]] = blobs_appr[:, [6, 5]]   # swap params

                        st.session_state['detected'] = True              
                        st.session_state['BLOBs_data'] = blobs_appr
                        st.session_state['detectParticles'] = detectionParticles
                        st.session_state['detectedParticles'] = blobs_appr.shape[0]
                        st.session_state['detectionSettings'] = [
                            st.session_state['typeImg'],
                            st.session_state['param-pre-1'],
                            st.session_state['param-pre-2'],
                            #st.session_state['param-pre-3']
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
                            help = tooltips.Filtration.Brightness
                        )

                        # Settings slider with diameters
                        tmp_diameters = st.session_state['scale'].apply(st.session_state['BLOBs_data'][:, 2])
                        min_d = np.min(tmp_diameters)
                        max_d = np.percentile(tmp_diameters, 98)

                        slider_min = np.floor(min_d / 10) * 10
                        slider_max = np.ceil(max_d / 10) * 10

                        if slider_max <= slider_min:
                            slider_max = slider_min + 10

                        if ('param-filt-2' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-2'] = (min_d, max_d)

                        st.slider(f"Nanoparticles diameter, {st.session_state['scale'].unit}",
                            key = 'param-filt-2',                         
                            min_value = slider_min,
                            step = 0.1,
                            max_value = slider_max,
                            format = "%0.1f",
                            disabled = st.session_state['settingDefault'],
                            help = tooltips.Filtration.Diameter 
                                   + f". {np.sum(tmp_diameters > max_d)} particles exceed the 98th percentile with a diameter greater than {slider_max}"
                        )

                        if ('param-filt-3' not in st.session_state) or st.session_state['settingDefault']:
                            st.session_state['param-filt-3'] = 0.75

                        st.slider("Nanoparticle reliability",
                            key = 'param-filt-3',
                            min_value = 0.0,
                            step = 0.01,
                            max_value = 1.0,
                            disabled = st.session_state['settingDefault'],
                            help = tooltips.Filtration.Reliability
                        )
                        
                        ### [Deprecated]
                        # if st.session_state['param-pre-3']:
                        #     if ('param-filt-4' not in st.session_state) or st.session_state['settingDefault']:
                        #         st.session_state['param-filt-4'] = 2500

                        #     st.slider("Area of background irregularities",
                        #         key = 'param-filt-4',
                        #         min_value = 0,
                        #         step = 25,
                        #         max_value = 5000,
                        #         disabled = st.session_state['settingDefault'],
                        #         help = tooltips.Filtration.Irregularities
                        #     )

                        #     temp_img = ExpApp.PreprocessingMedian(st.session_state['srcImg'].copy(), 3)
                        #     temp_img = ExpApp.PreprocessingTopHat(temp_img, 9)   

                        #     _, img_contours, st.session_state['big_contours'] = ExpApp2.FindAreasToDelete(temp_img, 85, st.session_state['param-filt-4'])
                        #     temp_BLOBs_data = ExpApp2.DeleteBorderPointsM(st.session_state['BLOBs_data'], img_contours, 255)
                     
            
                    # Filtering
                    BLOBs_data_filt = ExpApp.my_FilterBlobs_change(
                        st.session_state['BLOBs_data'], # if not st.session_state['param-pre-3'] else temp_BLOBs_data,
                        {
                            "thr_c0": st.session_state['param-filt-1'],
                            "min_thr_d": st.session_state['param-filt-2'][0] * st.session_state['scale'].divider,   
                            "max_thr_d": st.session_state['param-filt-2'][1] * st.session_state['scale'].divider, 
                            "thr_error": 1 - st.session_state['param-filt-3'], 
                        }
                    )

                    st.session_state['filterParticles'] = rEA.filtrationParticles(
                        st.session_state['detectParticles'],
                        c0 = (st.session_state['param-filt-1'], None),
                        diameter = (
                            st.session_state['param-filt-2'][0] * st.session_state['scale'].divider,
                            st.session_state['param-filt-2'][1] * st.session_state['scale'].divider
                        ),
                        approxError = (None, 1 - st.session_state['param-filt-3'])
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
                        st.toggle("Estimated scale",
                            key = 'displayScale', 
                            disabled = True,
                            help = tooltips.Visualization.Scale
                        )
                        
                        if (st.session_state['displayScale'] and st.session_state['scaleInfo'] is None):
                            st.warning(tooltips.Warnings.OutScale, icon = ":material/warning:") 

                        ### [Deprecated]
                        # Highlighting background irregularities
                        # st.toggle("Highlighting background irregularities",
                        #     key = 'areas',
                        #     disabled = not st.session_state['param-pre-3'],
                        #     help = tooltips.Visualization.Irregularities
                        # )
                            
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
                        fileResultName = Path(uploadedImg.name).stem
                        button_download_disabled = False

                        match selectionSave:
                            case 0:
                                temp = Image.new(mode = "RGBA", size = st.session_state['sizeImage'])
                                draw = ImageDraw.Draw(temp)
                                for BLOB in st.session_state['BLOBs_filter']:                
                                    y, x, d = BLOB; r = d/2          
                                    draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                temp.save(fileResult, format = 'png')
                                fileResultName += f"_particles.tif"

                            case 1:
                                imgBLOB = st.session_state['srcImg'].convert("RGB")
                                draw = ImageDraw.Draw(imgBLOB)                            
                                for BLOB in st.session_state['BLOBs_filter']:                
                                    y, x, d = BLOB; r = d/2
                                    draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                imgBLOB.save(fileResult, format = 'png')
                                fileResultName += f"_particls+image.tif"

                            case 2:
                                fileResult = io.StringIO()
                                temp_writer = csv.writer(fileResult, delimiter = ';')
                                temp_writer.writerow([f"Scale: {st.session_state['scale'].multiplier:.3} ({st.session_state['scale'].unit}/px)"])
                                temp_writer.writerow(['coord y, px', 'coord x, px', 'diameters, px'])
                                temp_writer.writerows(st.session_state['BLOBs_filter'])
                                fileResultName += f"_parameters.csv"
                            case 3:
                                imageData= {
                                    'name': Path(uploadedImg.name).stem,
                                    'width': st.session_state['srcImg'].size[0],
                                    'height': st.session_state['srcImg'].size[1],
                                    'buffer': uploadedImg.getvalue()
                                }
                                fileResult = API2CVAT.ExportToCVAT(imageData, st.session_state['BLOBs_filter'])
                                fileResultName += f"_{time.strftime('%Y-%m-%d-%H-%M-%S')}.zip"
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
         
            
            # Display image 
            with colImage:
                particles = []
                if st.session_state['filterParticles'] is not None:
                    particles = [
                        particle.toDict() for particle in st.session_state['filterParticles']
                    ]          

                streamlit_image_overlay(
                    image = st.session_state["srcImg"],
                    overlays = particles,
                    key = "main-imageViewer"                    
                )



    ## TAB 2 
    with tabStat:    
        heightCol = 550
        marginChart = dict(l=10, r=10, t=10, b=5)
        #marginChartLess = dict(l=5, r=5, t=10, b=5)
              
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

                        #TO DO: fix resize
                        st.session_state['statImage'] =  st.session_state['statImage'].resize((1280, 960))

                        #TO DO: The global scale is used to automatically detect and download the CVAT.
                        #  It's working now because the first layout is always executed and the image scale
                        #  is always used there. and here it works because the scale is used from a backup.
                        #  It is better to reduce these scales.
                        tmp_scale, lowerBound, st.session_state['scaleInfo'] = analyzeScaleRegion(st.session_state['statImage'].convert("L"))
                        st.session_state['scale'] = autoscale.Scale(tmp_scale)

                        st.session_state['sizeImage'] = list(st.session_state['statImage'].size)
                        if lowerBound is not None:
                            st.session_state['sizeImage'][1] = lowerBound
                case _:
                    st.session_state['calcStatictic'] = False

        if (not st.session_state['calcStatictic']):
            defaultStatTab()
        else:
            with st.expander("Particle parameters", expanded = True, icon = ":material/app_registration:"):
                instruct.AboutSectionParticleParams()
                                
                boolIndexSelectedBLOBs = None   

                diameter_nm = st.session_state['scale'].apply(st.session_state['statBLOBs'][:, 2])
                BLOBs_nm = st.session_state['scale'].apply(st.session_state['statBLOBs'])

                db11, db12, db13 = st.columns([4, 4, 4])            

                # Particle size distribution
                with db11.container(border = True, height = heightCol):
                    with st.popover("Distribution of particle diameters", width = 'stretch'):
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
                                                                      
                    step = st.session_state['step']
                    start = np.floor(diameter_nm.min()) - step
                    end = np.ceil(diameter_nm.max()) + step

                    counts, bins = np.histogram(diameter_nm, bins = np.arange(start, end, step, dtype = float))
                                        
                    name_x = f"Diameters, {st.session_state["scale"].unit}"
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
                        width = 'stretch',
                        help = tooltips.Distribution.Download
                    )

                    fig = go.Figure()

                    fig = fig.add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = bar_y,
                        customdata = customDataChart,
                        showlegend = False,
                        hovertemplate = (
                            f"Diameter: [%{{customdata[0]:.1f}}, %{{customdata[1]:.1f}}) {st.session_state["scale"].unit}<br>"
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
                        margin = marginChart,
                        xaxis_title_text = name_x,
                        yaxis_title_text = name_y,                        
                        bargap = 0,
                        legend = dict(
                            x = 1,
                            y = 1,
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
                        width = 'stretch',
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
                    with st.popover("Nanoparticle parameters", width = 'stretch'):
                        selectionMaterial = st.pills(
                            "Particles material",
                            default = 0,
                            required = True,
                            width = 400,
                            options = tooltips.Options.MaterialName.keys(),
                            format_func = lambda option: tooltips.Options.MaterialName[option],
                        )

                        if selectionMaterial is None:
                            selectionMaterial = 0

                        materialName = tooltips.Options.MaterialName[selectionMaterial]

                        if selectionMaterial == 4:
                            materialDensity = st.number_input(
                                "Particles material density on ng/nm³",
                                min_value = 0.0,
                                step = 1.0e-11,
                                value = 1.0e-10,
                                format = "%0.2e",
                                key = "user-density"
                            )
                        else:
                            materialDensity = tooltips.Options.MaterialDensity[selectionMaterial]

                        instruct.MaterialDensity(materialName, materialDensity)
                    
                    # Additional info                                     
                    instruct.EstimatedScale(st.session_state["scale"]) # TODO input scale
                    
                    if selectionMaterial == 4: # User material
                        instruct.UserMaterial(materialName, materialDensity)
                    else:
                        instruct.DefMaterial(materialName)
                    
                    currentDiameter = diameter_nm
                    currentBLOBs = BLOBs_nm
                    if boolIndexSelectedBLOBs is not None:
                        currentDiameter = currentDiameter[boolIndexSelectedBLOBs]
                    
                    paramsNP = NanoStat.calculateParametersNP(
                        currentDiameter,
                        materialDensity,
                        st.session_state['sizeImage'],
                        st.session_state['scale'].multiplier # TODO add st.session_state with key 'areaImage'
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

                # Visualization particles
                with db13.container(border = True, height = heightCol):
                    with st.popover("Visualization particles", width = 'stretch'):                    
                        tempSelectionChart = st.pills(
                            "Type visualization",
                            default = 1,
                            required = True,
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
                                hovertemplate = "Particle in subarea %{z:.2}<extra></extra>"
                            )

                            fig.update_layout(
                                margin = marginChart,
                                xaxis_title_text = f'Width image, {stepSize}*px',
                                yaxis_title_text = f'Height image, {stepSize}*px',
                                coloraxis_colorbar = dict(
                                    title = "Particle count",
                                    orientation = "h",
                                    y = -0.2,
                                ),
                                showlegend = False
                            )

                            st.plotly_chart(fig, width = 'stretch',)
                        case 1: 
                            currentBLOBs = st.session_state['statBLOBs']
                            if boolIndexSelectedBLOBs is not None:         
                                currentBLOBs = currentBLOBs[boolIndexSelectedBLOBs]

                            tempImage = st.session_state['statImage'].copy()
                            draw = ImageDraw.Draw(tempImage)                            
                            for BLOB in currentBLOBs:                
                                y, x, d = BLOB; r = d/2
                                draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)
                            
                            st.image(tempImage, width = 'stretch',)
                            # streamlit_image_overlay(
                            #     image = tempImage,
                            #     overlays = [],
                            #     key = "bd13-imageViewer"                    
                            # )

                # END db13

            with st.expander("Nanoparticle spatial distribution", expanded = True, icon = ":material/data_thresholding:"):
                instruct.AboutSectionSpatialDistribution()

                currentBLOBs = st.session_state['scale'].apply(np.copy(st.session_state['statBLOBs']))
                fullDist, minDist = NanoStat.euclideanDistance(currentBLOBs) 

                db21, db22, db23 = st.columns([1, 1, 1])
                
                # Fraction of empty subareas
                with db21.container(border = True, height = heightCol):  
                    with st.popover("Fraction of empty subareas", width = 'stretch'):
                        # Saving raw data db21
                        db21_buttonPlaceholder = st.empty()

                    x = np.arange(5, 105, 5)
                    emptySubareas = np.zeros_like(x, dtype = float)
                    emptyCount = np.zeros_like(x, dtype = int)
                    totalCount = np.zeros_like(x, dtype = int)

                    for i, size in enumerate(x):
                        temp = NanoStat.uniformity(
                            st.session_state["statBLOBs"],
                            st.session_state["sizeImage"],
                            size
                        )
                        emptyCount[i] = np.sum(temp == 0)
                        totalCount[i] = temp.size

                        emptySubareas[i] = emptyCount[i] / totalCount[i]                    
                    
                    fig = px.bar(x = st.session_state['scale'].apply(x), y = emptySubareas)

                    fig.update_layout(
                        margin = marginChart,
                        xaxis_title_text = f'Size of square subareas, {st.session_state["scale"].unit}',
                        yaxis_title_text = 'Empty subareas fraction',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hovertemplate = f"Size: %{{x:.2}} {st.session_state["scale"].unit} <br>Empty: %{{y:.2}}",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 1
                    )

                    st.plotly_chart(fig, width = 'stretch',)

                    # Saving db21 
                    db21_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Block size ({st.session_state['scale'].unit})": st.session_state['scale'].apply(x),
                            "Empty subareas": emptyCount,
                            "Total subareas": totalCount,
                            "Empty fraction": emptySubareas,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['statImageName']}-empty-subareas.csv",
                        width = 'stretch', 
                        icon = ":material/download:",
                        help = tooltips.NoneInfo
                    )
                # END db21

                # Distance to nearest nanoparticle
                with db22.container(border = True, height = heightCol):
                    with st.popover("Distance to nearest nanoparticle", width = 'stretch'):                        
                        # Saving raw data db22
                        db22_buttonPlaceholder = st.empty()

                    counts, bins = np.histogram(minDist, bins = np.arange(0, 50, 2, dtype = float))                      
                    distanceNearest = counts / np.sum(counts)

                    fig = go.Figure()

                    fig = fig.add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = distanceNearest,
                        showlegend = False,
                    ))
                        
                    fig.update_layout(
                        margin = marginChart,                        
                        bargap = 0,
                        xaxis_title_text = f'Distance to nearest nanoparticle, {st.session_state["scale"].unit}',
                        yaxis_title_text = 'Particle fraction',
                        showlegend = False
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hovertemplate = f"Distanse: %{{x:.2}} {st.session_state["scale"].unit} <br>Fraction: %{{y:.2}}<extra></extra>",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 1,  
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                                        
                    # Saving db22
                    db22_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Distance ({st.session_state["scale"].unit})": minDist,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['statImageName']}-distance-nearest-nanoparticle.csv",
                        width = 'stretch', 
                        icon = ":material/download:",
                        help = tooltips.NoneInfo
                    )
                # END db22

                # Average number of nanoparticles per unit area
                with db23.container(border = True, height = heightCol):
                    with st.popover("Nanoparticles per unit area", width = 'stretch'):
                         # Saving raw data db23
                        db23_buttonPlaceholder = st.empty()
                    
                    # TODO: what the scale of 'x'?
                    x = st.session_state["scale"].apply(np.arange(5, 105, 1))

                    averageDensity = NanoStat.averageDensityInNeighborhood(x, fullDist)
                    numberLess = np.rint(averageDensity * np.pi * x**2 * len(fullDist)).astype(int)
                                                            
                    fig = px.bar(x = x , y = averageDensity)

                    fig.update_layout(
                        margin = marginChart,
                        xaxis_title_text = f'Neighborhood radius, {st.session_state["scale"].unit}',
                        yaxis_title_text = f'Nanoparticles per unit area, particles/{st.session_state["scale"].unit}²',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hovertemplate = f"Neighborhood radius: %{{x:.2}} {st.session_state["scale"].unit} <br>Particles/{st.session_state["scale"].unit}²: %{{y:.1e}}",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 0.5
                    )

                    st.plotly_chart(fig, width = 'stretch')
                    
                    # Saving db23
                    db23_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Neighborhood radius ({st.session_state["scale"].unit})": x,
                            "Number of particles": len(fullDist),
                            "Number of neighbors": numberLess,
                            f"Average density (particles/{st.session_state["scale"].unit}^2)": averageDensity,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['statImageName']}-nanoparticles-unit-area.csv",
                        width = 'stretch', 
                        icon = ":material/download:",
                        help = tooltips.NoneInfo
                    )
                # END db23

                
                db31, db32, db33 = st.columns([1, 1, 1])

                # Average coverage of neighbors
                with db31.container(border = True, height = heightCol): 
                    with st.popover("Average coverage of neighbors", width = 'stretch'):
                        pass   

                    x = st.session_state["scale"].apply(np.arange(10, 105, 1))
                    localArea = NanoStat.localAreaFraction(x, fullDist, currentBLOBs[:, 2]) * 100
                    
                    fig = px.bar(x = x, y = localArea)

                    fig.update_layout(
                        margin = marginChart,
                        xaxis_title_text = f'Neighbors area size, {st.session_state["scale"].unit}',
                        yaxis_title_text = 'Neighbors coverage, %',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hovertemplate = f"Area size: %{{x:.2}} {st.session_state["scale"].unit} <br>Coverage: %{{y:.3}}%",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 0.5
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                # END db31

                # Average number of neighbors
                with db32.container(border = True, height = heightCol):  
                    with st.popover("Average number of neighbors", width = 'stretch'):
                        pass               
                    
                    x = st.session_state["scale"].apply(np.arange(10, 105, 1))
                    averageNeighborhoods = NanoStat.averageNeighborhoods(x, fullDist)
                    
                    fig = px.bar(x = x, y = averageNeighborhoods)

                    fig.update_layout(
                        margin = marginChart,
                        xaxis_title_text = f'Nanoparticle neighborhood size, {st.session_state["scale"].unit}',
                        yaxis_title_text = 'Average number of neighbors',
                        showlegend = False,
                        bargap = 0
                    )
                    
                    fig.update_xaxes(
                        showgrid = True,
                    )

                    fig.update_traces(
                        hovertemplate = f"Size: %{{x:.2}} {st.session_state["scale"].unit} <br>Neighbors: %{{y:.2}}",
                        marker_color = colorRGBA_str,
                        marker_line_color = 'blue',
                        marker_line_width = 0.5
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                # END db32
            
                # Statistics aggregator
                # TODO: make it readable
                with db33.container(border = True, height = heightCol):
                    st.subheader("Aggregate statistics", width = 'stretch') 

                    # check the presence of this image in table
                    if st.session_state["statImageName"] != st.session_state.get("previousImage"):
                        st.session_state["previousImage"] = st.session_state["statImageName"] 
                        if not st.session_state["analysisBuffer"]["Image"].eq(st.session_state["statImageName"]).any():
                            if len(minDist) < 3:
                                mp_nearest = np.mean(minDist)
                            elif np.std(minDist) == 0:
                                mp_nearest = minDist[0]
                            else:
                                kde = scipy.stats.gaussian_kde(minDist)
                                kde_x = np.linspace(
                                    minDist.min(),
                                    minDist.max(),
                                    1000
                                )
                                density = kde(kde_x)
                                mp_nearest = kde_x[np.argmax(density)]
                        
                            mean_nearest = np.mean(minDist)
                            threshold_nm = round(np.mean(currentBLOBs[:, 2]))
                            area_nm2 = np.prod(st.session_state["sizeImage"]) * st.session_state['scale'].multiplier**2
                            cl_ev = 2 * mean_nearest * np.sqrt(len(currentBLOBs) / area_nm2)

                            newRow = {
                                "Image": st.session_state["statImageName"],
                                "Number of particles": len(currentBLOBs),
                                "Material type": materialName,
                                "Mean particle diameter, nm": np.mean(currentBLOBs[:, 2]),
                                "Particle surface density, mg/m²": 
                                    np.sum((1/6 * np.pi * currentBLOBs[:, 2]**3) * materialDensity * 10**+12) / area_nm2,
                                "Mean distance to neighbour, nm": mean_nearest,
                                "Most probable distance to neighbour, nm": mp_nearest,
                                "Distance threshold, nm": threshold_nm,
                                "Fraction below distance threshold": np.sum(minDist < threshold_nm) / len(currentBLOBs),
                                "Clark-Evans index (R)": cl_ev,
                            }
                        
                            st.session_state["analysisBuffer"] = pd.concat(
                                [ st.session_state["analysisBuffer"], pd.DataFrame([newRow]) ],
                                ignore_index = True
                            )
                   
                    buffer = st.session_state["analysisBuffer"].copy()
                    buffer.insert(0, "Delete", False)

                    editedBuffer = st.data_editor(
                        buffer,
                        width = "stretch",
                        num_rows = "fixed",
                        hide_index = True,
                        disabled = [
                            col for col in buffer.columns
                            if col != "Delete"
                        ],
                        column_config = {
                            "Delete": st.column_config.CheckboxColumn(
                                "Delete",
                                width = "small",
                            ),
                        },
                    )

                    deleteMask = editedBuffer["Delete"].astype(bool)

                    if deleteMask.any():
                        st.session_state["analysisBuffer"] = (
                            editedBuffer.loc[~deleteMask]
                                .drop(columns = "Delete")
                                .reset_index(drop = True)
                        )
                        st.rerun()  # !!!
                # END db33

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


                                temp_gt_blobs = st.session_state["scale"].apply(gt_blobs[:, 2])
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

                                temp_ALL = st.session_state["scale"].apply(ALL[:, 2])
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

                                st.plotly_chart(fig, width = 'stretch',)


    ## TAB 3
    with tabHelp:
        if st.button("If you have any difficulties with our tool, please contact us (click here)",
            key = 'button_contact',
            width = 'stretch',
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