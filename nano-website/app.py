# Run application
# streamlit run .\nano-website\app.py --server.enableXsrfProtection false

import streamlit as st

import io, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import time, datetime

import style, autoscale
import NanoStatistics as NanoStat
import ExponentialApproximation as ExpApp
import CustomComponents as CustComp
import WebsiteBot as webBot

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import traceback
    

### Function ###
    
help_str = "hint be added soon"

colorRGBA_str = 'rgb(150, 150, 255)'
colorRGB = (150, 150, 255)


def load_default_session_state(_dispToast = False):
    if _dispToast:
        st.toast('Default configuration loaded!')

    st.session_state['rerun'] = False

    st.session_state['imgUpload'] = False
    st.session_state['uploadedImg'] = None
    st.session_state['srcImg'] = None
    st.session_state['typeImg'] = None
    st.session_state['fileImageName'] = None

    st.session_state['imgPlaceholder'] = None

    st.session_state['settingDefault'] = True
    st.session_state['param-pre-1'] = 10
    st.session_state['param1'] = 10
    st.session_state['param2'] = (0.5, 10.0)
    st.session_state['param3'] = 0.65
    st.session_state['parallel'] = True
    st.session_state['processes'] = 6
            
    st.session_state['detected'] = False
    st.session_state['BLOBs'] = None
    st.session_state['BLOBs_params'] = None
    st.session_state['BLOBs_filter'] = None
    st.session_state['sizeImage'] = None
    st.session_state['detectedParticles'] = 0    
    st.session_state['filteredParticles'] = 0 
    st.session_state['imgBLOB'] = None
    st.session_state['timeDetection'] = None
    st.session_state['detectionSettings'] = None

    st.session_state['comparison'] = False
    
    st.session_state['scale'] = None
    st.session_state['scaleData'] = None
    st.session_state['displayScale'] = False

    st.session_state['distView'] = False
    st.session_state['normalize'] = False
    st.session_state['selection'] = False

    st.session_state['calcStatictic'] = False

    #st.session_state['equalize'] = False
    #st.session_state['invers'] = True    
    #st.session_state['median'] = None    
    #st.session_state['top-hat'] = None    
    st.session_state['reprocess'] = False


def session_state2str(closedKey = ["imgPlaceholder", ]):
    tempStr = "\n"
    for key in st.session_state.keys():
        if key not in closedKey:
            tempStr = tempStr + f"\t{key}: {str(st.session_state[key])}\n"

    return tempStr


@st.dialog("Something went wrong...")
def dialog_exception(_exception):
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
        data = {
            "dump": session_state2str(),
            "traceback": traceback.format_exc(),    
            "image-data": None,
            "image-type": None
        }

        if st.session_state['uploadedImg'] is not None:
            data.update({                
                "image-data": st.session_state['uploadedImg'].getvalue(),
                "image-type": st.session_state['uploadedImg'].type
            })       

        result, response = webBot.message2email(data)

    with st.expander("Info for developers", expanded = False, icon = ":material/app_registration:"):
        st.error(traceback.format_exc())
        
        if result:
            st.success("Report successful sent!")
        else:
            st.error("Error sending report: " + str(response.json()))

def update_calcStatictic():                
    st.session_state['calcStatictic'] = True

def update_detected():
    st.session_state['detected'] = True



### Main app ###
try:
    # Loading CSS styles
    st.set_page_config(page_title = "Nanoparticles", layout = "wide")
    style.set_style(colorRGBA_str)
    
    # Initial loading of session states
    if 'rerun' not in st.session_state:
        load_default_session_state()
    elif st.session_state['rerun']:
        load_default_session_state(True)
    
    ## Header
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
    
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
            <br>It will help you to detect palladium nanoparticles in the image and calculate their statictics.
        </div>
    """, unsafe_allow_html = True)

    st.markdown("""
        <div style = "padding-bottom: 25px" class = 'about'>
            Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
        </div>
    """, unsafe_allow_html = True)


    ## Main content area
    tabDetect, tabInfo, tabLable, tabGuide  = st.tabs([
        "Automatic detection nanoparticles",
        "Statistics dashboard",
        "Manual labeling nanoparticles",
        "User's Guide"
    ])

    ## TAB 1
    with tabDetect:
        imgPlaceholder = None
        
        st.subheader("Upload SEM image")            
        uploadedImg = st.file_uploader("Choose an SEM image", type = ["tif", "tiff", "png", "jpg", "jpeg" ])
        st.session_state['uploadedImg'] = uploadedImg

        if uploadedImg is None:
            load_default_session_state()
        else:
            st.session_state['imgUpload'] = True   
            if (st.session_state['fileImageName'] != uploadedImg.name):
                load_default_session_state()
                
                srcImage = Image.open(uploadedImg).convert("L")
                #srcImage = srcImage.resize((1280, 960))    
                
                st.session_state['srcImg'] = srcImage
                st.session_state['fileImageName'] = uploadedImg.name
            else:
                srcImage = st.session_state['srcImg']
        
        if (st.session_state['imgUpload']):
            colImage, colSetting = st.columns([6, 2])

            with colImage:
                if (st.session_state['imgPlaceholder'] is None):
                    st.session_state['imgPlaceholder'] = st.empty()

                if (st.session_state['imgBLOB'] is not None):
                    if (not st.session_state['comparison']):
                        st.session_state['imgPlaceholder'].image(st.session_state['imgBLOB'], use_container_width = True)
            # END left side

            # Detection settings and results
            with colSetting:        
                st.checkbox("Use default settings?",
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
                        params = {
                            # размер окна медианного фильтра
                            "sz_med" : 4,
                            # размер диска Top-Hat (не надо равное 5 - кружки получаются большие) 
                            "sz_th":  6,
                        }
                    elif (st.session_state['typeImg'] == 'SEM'):
                        params = {
                            # размер окна медианного фильтра
                            "sz_med" : 4,
                            # размер диска Top-Hat (не надо равное 5 - кружки получаются большие) 
                            "sz_th":  4,
                        }

                    tempArrImg = np.array(srcImage, dtype = 'uint8')
                    tempArrImg = ExpApp.PreprocessingMedian(tempArrImg, params['sz_med'])
                    tempArrImg = ExpApp.PreprocessingTopHat(tempArrImg, params['sz_th'])

                    srcImage = Image.fromarray(tempArrImg, mode = 'L')
                            
                # Detection settings       
                with st.expander("Detection settings", expanded = not st.session_state['detected'], icon = ":material/tune:"):
                    st.slider("Nanoparticle brightness",
                        key = 'param-pre-1',
                        disabled = st.session_state['settingDefault'],
                        help = "The average brightness of nanoparticles and its surroundings in the image"
                    )

                    l, r = st.columns([3,1])
                    l.checkbox("Parallel computing", key = 'parallel', disabled = st.session_state['settingDefault'])

                    r.number_input(label = "not_visibility",
                        min_value = 1,
                        max_value = 8,
                        step = 1,
                        format = "%i",
                        placeholder = "Processes",
                        key = 'processes',
                        disabled = st.session_state['settingDefault'],
                        label_visibility = 'collapsed'
                    )

                    warningPlaceholder = st.empty()
                    if (st.session_state['detectionSettings'] is not None) and st.session_state['detected']:
                        if (st.session_state['detectionSettings'] != [st.session_state['param-pre-1'], st.session_state['parallel']]):
                            warningPlaceholder.warning("""
                                The detection settings have been changed.
                                To accept the new settings, click the button "Nanoparticles detection"! 
                            """, icon = ":material/warning:")
        
                    pushDetectButton = st.button("Nanoparticles detection",
                        use_container_width = True,
                        disabled = not st.session_state['imgUpload'],
                        help = help_str,
                        on_click = update_detected
                    )
                
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
                        
                        # параметры в пикселях
                        params = {
                            # порог яркости для отбрасывания лок. максимумов
                            "thr_br": float(st.session_state['param-pre-1']),   
                            # минимальное расстояние между локальными максимумами при их поиске 
                            "min_dist": 4,
                            # размер окна аппроксимации
                            "wsize": 9,        
                            # возможные радиусы наночастиц в пикселях
                            "rs": np.arange(0.5, 6.5, 0.1), 
                            # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                            "best_mode": 3, 
                            # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                            "msk": 3,      
                            # аппроксимирующая функция "exp" или "pol" 
                            "met": 'exp',   
                            # число параметров аппроксимации
                            "npar": 2       
                        }

                        # вычисляется только один раз при первом запуске детектирования
                        helpMatrs, xy2 = ExpApp.CACHE_HelpMatricesNew(params["wsize"], params["rs"])

                        # вычисляется только один раз для одного порога яркости
                        lm, _ = ExpApp.CACHE_PrefilteringPoints(
                            currentImage,
                            params,
                            False,
                            False
                        )

                        # вычисляется только один раз для одного набора параметров
                        if st.session_state['parallel']:
                            BLOBs, BLOBs_params = ExpApp.CACHE_ExponentialApproximationMask_v3_parallel(
                                currentImage,
                                lm,
                                xy2,
                                helpMatrs,
                                params,
                                nProc = st.session_state['processes']
                            )
                        else:
                            BLOBs, BLOBs_params = ExpApp.CACHE_ExponentialApproximationMask_v3(
                                currentImage,
                                lm,
                                xy2,
                                helpMatrs,
                                params
                            )

                        st.session_state['detected'] = True                
                        st.session_state['BLOBs'] = BLOBs
                        st.session_state['BLOBs_params'] = BLOBs_params
                        st.session_state['detectedParticles'] = BLOBs.shape[0]
                        st.session_state['detectionSettings'] = [st.session_state['param-pre-1'], st.session_state['parallel']]
        
                    st.session_state['timeDetection'] = int(np.ceil(time.time() - timeStart))

                # Detection results
                if st.session_state['detected']:
                    time = st.session_state['timeDetection']
                    st.markdown(f"""
                        <p class = 'text'>
                            Nanoparticles detected: <b>{st.session_state['detectedParticles']}</b> ({time//60}m : {time%60:02}s)
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
                        st.slider("Nanoparticle center brightness",
                            key = 'param1',
                            value = 10,
                            disabled = st.session_state['settingDefault'],
                            help = "Brightness in the central pixel of the nanoparticle"
                        )

                        temp_max_r_nm = 10.0
                        temp_min_r_nm = 1
                        if (st.session_state['BLOBs'] is not None) and (st.session_state['scale'] is not None):
                            temp_max_r_nm = np.max(st.session_state['BLOBs'][:, 2]) * st.session_state['scale']
                            temp_min_r_nm = np.min(st.session_state['BLOBs'][:, 2]) * st.session_state['scale']

                        st.slider("Range of nanoparticle diameter, nm",
                            key = 'param2',
                            #value = (0.5, 10.0),
                            min_value = np.floor(temp_min_r_nm-1),
                            step = 0.1,
                            max_value = np.ceil(temp_max_r_nm+1),
                            format = "%0.1f",
                            disabled = st.session_state['settingDefault']
                        )

                        st.slider("Nanoparticle reliability",
                            key = 'param3',
                            value = 0.65,
                            min_value = 0.0,
                            step = 0.01,
                            max_value = 1.0,
                            disabled = st.session_state['settingDefault'],
                            help = "The higher the reliability, the clearer the nanoparticle is against the background of the image"
                        )
                
                    divider = 1
                    if st.session_state['scale'] is not None:
                        divider = st.session_state['scale']

                    params_filter = {
                        "thr_c0": st.session_state['param1'],
                        "min_thr_d": st.session_state['param2'][0] / divider,   
                        "max_thr_d": st.session_state['param2'][1] / divider, 
                        "thr_error": 1 - st.session_state['param3'], 
                    }
            
                    # Filtering
                    st.session_state['BLOBs_filter'], _ = ExpApp.my_FilterBlobs_change(
                        st.session_state['BLOBs'],
                        st.session_state['BLOBs_params'],
                        params_filter
                    )
                    st.session_state['filteredParticles'] = st.session_state['BLOBs_filter'].shape[0]

                    if (st.session_state['filteredParticles'] < 1):
                        st.warning("""
                            There are no nanoparticles satisfying the filtration settings!
                            Please change the filtering settings!
                        """, icon = ":material/warning:")
                    else:                    
                        # Info about filtered nanoparticles
                        st.markdown(f"""
                            <p class = 'text'>
                                Nanoparticles after filtration: <b>{st.session_state['filteredParticles']}</b>
                            </p>""", unsafe_allow_html = True
                        )
                                        
                        with st.expander("Visualization and saving results", expanded = True, icon = ":material/display_settings:"):
                            # Displaying the scale
                            st.toggle("Estimated scale display", key = 'displayScale', disabled = True, help = help_str)

                            if (st.session_state['displayScale'] and st.session_state['scaleData'] is None):
                                st.warning("""
                                    The image scale could not be determined automatically!
                                    Using default scale: 1.0 nm/px
                                """, icon = ":material/warning:")                    

                            # Slider for comparing the results before and after detection
                            st.toggle("Comparison mode", key = 'comparison', help = help_str)

                            # 
                            st.toggle("Display preprocessing image", key = 'reprocess', disabled = False, help = help_str)

                            # Saving
                            selectboxCol, buttonCol = st.columns([6,1], vertical_alignment = 'bottom')

                            option_saving = {
                                0: "Particles on clear background (*.tif)",
                                1: "Particles on EM-image (*.tif)",
                                2: "Particles characteristics (*.csv)",
                            }

                            selection = selectboxCol.selectbox(
                                "What results should be saved?",
                                index = 1,
                                placeholder = "Select options...",
                                options = option_saving.keys(),
                                format_func = lambda option: option_saving[option]
                            )

                            fileResult = io.BytesIO()
                            fileResultName = 'None'
                            button_download_disabled = False

                            match selection:
                                case 0:
                                    temp = Image.new(mode = "RGBA", size = st.session_state['sizeImage'])
                                    draw = ImageDraw.Draw(temp)
                                    for BLOB in st.session_state['BLOBs_filter']:                
                                        y, x, d = BLOB; r = d/2          
                                        draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                    temp.save(fileResult, format = 'png')
                                    fileResultName = "particls-" + Path(uploadedImg.name).stem + ".tif"

                                case 1:
                                    imgBLOB = st.session_state['srcImg'].convert("RGB")
                                    draw = ImageDraw.Draw(imgBLOB)                            
                                    for BLOB in st.session_state['BLOBs_filter']:                
                                        y, x, d = BLOB; r = d/2
                                        draw.ellipse((x-r, y-r, x+r, y+r), outline = colorRGB)

                                    imgBLOB.save(fileResult, format = 'png')
                                    fileResultName = "particls+image-" + Path(uploadedImg.name).stem + ".tif"

                                case 2:
                                    fileResult = io.StringIO()

                                    temp_writer = csv.writer(fileResult, delimiter = ';')
                                    if st.session_state['scale'] is not None: 
                                        temp_writer.writerow(["Scalse:", f"{st.session_state['scale']:.3}", "nm/px"])
                                    else:
                                       temp_writer.writerow([f"Using default scale:", "1.0", "nm/px"])
                    
                                    temp_writer.writerow(['coord y, px', 'coord x, px', 'diameters, px'])
                                    temp_writer.writerows(st.session_state['BLOBs_filter'])

                                    fileResultName = "particls_info-" + Path(uploadedImg.name).stem + ".csv"

                                case _:
                                    button_download_disabled = True


                            buttonCol.download_button(
                                label = "",
                                icon = ":material/download:",
                                data = fileResult.getvalue(),
                                file_name = fileResultName,
                                disabled = button_download_disabled
                            )
            # END right side

        # Display source image by st.image
        if (st.session_state['imgUpload']):
            viewImage = st.session_state['srcImg'].copy().convert('RGB')

            if (st.session_state['reprocess']):
                viewImage.paste(srcImage, (0,0))

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

            st.session_state['imgBLOB'] = viewImage

            if (st.session_state['comparison']):
                with st.session_state['imgPlaceholder'].container():
                    if (st.session_state['reprocess']):
                        temp = st.session_state['srcImg'].copy().convert('RGB')
                        temp.paste(srcImage, (0,0))

                        CustComp.img_box(
                            temp,
                            st.session_state['imgBLOB'],
                            st.session_state['srcImg'].size                        
                        )
                    else:
                        CustComp.img_box(
                            st.session_state['srcImg'],
                            st.session_state['imgBLOB'],
                            st.session_state['srcImg'].size                        
                        )
            else:
                st.session_state['imgPlaceholder'].image(st.session_state['imgBLOB'], use_container_width = True)


    ## TAB 2
    with tabLable:
        st.warning("""
                The section is temporarily unavailable, but it will appear soon!
            """, icon = ":material/warning:")


    ## TAB 3
    with tabInfo:    
        heightCol = 550
        marginChart = dict(l=10, r=10, t=40, b=5)
        marginChartLess = dict(l=5, r=5, t=0, b=5)

        if (not st.session_state['detected']):
            st.session_state['calcStatictic'] = False

            st.warning("""
                Nanoparticle detection is necessary to calculate their statistics.
                Please go to "Automatic detection nanoparticles" tab.
            """, icon = ":material/warning:")
        elif (st.session_state['filteredParticles'] < 10):  
            st.session_state['calcStatictic'] = False
            
            st.warning("""
                Nanoparticles after detection and filtration are less than 10! 
                Please go to the "Detection" tab and change the detection,
                    filtering settings or upload another SEM image!
            """, icon = ":material/warning:")
        else:
            with st.expander("Global dashboard settings", expanded = not st.session_state['calcStatictic'], icon = ":material/rule_settings:"):

                option_map = {
                    0: "Automatically detected (AD)",
                    1: "Manual labeled (ML)",
                    2: "Union of AD and ML",
                    3: "Intersection of AD and ML",
                }

                selection = st.radio(
                    "Which nanoparticles to use?",
                    index = 0,
                    options = option_map.keys(),
                    format_func = lambda option: option_map[option],
                    horizontal = True,
                    disabled = True
                )                
          
                st.button("Calculate statistics", key = 'right_button', on_click = update_calcStatictic)           

        if (st.session_state['calcStatictic']):
            boolIndexSelectedBLOBs = None       
            
            with st.expander("Particle parameters", expanded = True, icon = ":material/app_registration:"):
                st.markdown(f"""
                    <p class = 'text'>
                        The main parameters of nanoparticles can be represented as primary values: 
                        the average diameters, its deviations, or a histogram of the diameters distribution. 
                        Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
                        which can be normalized to the area of the SEM image.
                    </p>""", unsafe_allow_html = True
                )
                
                diameter_nm = st.session_state['BLOBs_filter'][:, 2]  
                if st.session_state['scale'] is not None:
                    diameter_nm = diameter_nm * st.session_state['scale']

                db11, db12, db13 = st.columns([4, 4, 4])            

                # Particle size distribution
                with db11.container(border = True, height = heightCol):
                    left, rigth = st.columns([7, 1])
                    left.subheader("Distribution of particle diameters")

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
                        
                        buttonPlaceholder = st.empty()
                                        
                    start = st.session_state['param2'][0]
                    step = 0.2
                    end = st.session_state['param2'][1] + step

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
                        file_name = Path(uploadedImg.name).stem + "-dist-diameters.csv",
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
                    left.subheader("Nanoparticle parameters")

                    with rigth.popover("", icon=":material/settings:"):
                        option_materialDensity = {
                            0: "Palladium (Pd)",    # 12.02 * 10**-12 ng / nm^3
                            1: "Cuprum (Cu)",       #  8.96 * 10**-12 ng / nm^3
                            2: "Alloy 30% Au + 70% Pd (AuPd)",  # 14.10 * 10**-12 ng / nm^3
                            3: "Alloy 70% Cu + 30% Zn (CuZn)"   #  8.42 * 10**-12 ng / nm^3
                        }

                        selection = st.selectbox(
                            "Particles material",
                            index = 0,
                            placeholder = "Select options...",
                            options = option_materialDensity.keys(),
                            format_func = lambda option: option_materialDensity[option]
                        )

                        match selection:
                            case 0: materialDensity = 12.02 * 10**-12
                            case 1: materialDensity = 8.96 * 10**-12
                            case 2: materialDensity = 14.10 * 10**-12
                            case 3: materialDensity = 8.42 * 10**-12
                            case _: materialDensity = 1

                        st.markdown(f"""
                            <div class = 'text'>
                                Particles material density: <b>{materialDensity} ng/nm<sup>3</sup></b> 
                            </div>""", unsafe_allow_html = True)
                      
                    if st.session_state['scale'] is None:
                        pass
                        # st.warning("""
                        #     The image scale could not be determined automatically!
                        #     Using default scale: 1.0 nm/px
                        # """, icon = ":material/warning:")
                    else:                    
                        st.markdown(f"""
                            <div class = 'text'>
                                Estimated scale: <b>{st.session_state['scale']:.3f} nm/px</b> 
                            </div>""", unsafe_allow_html = True)
                    
                    st.markdown(f"""
                        <div class = 'text'>
                            Material: <b>{option_materialDensity[selection]}</b> 
                        </div>""", unsafe_allow_html = True)

                    temp_add_str = ""
                    currentDiameter = diameter_nm
                    if boolIndexSelectedBLOBs is not None:
                        currentDiameter = currentDiameter[boolIndexSelectedBLOBs]
                        temp_add_str = f"(includ {len(currentDiameter)} selected)"
   
                    st.markdown(f"""
                        <div class = 'text'>
                            Quantity: <b>{st.session_state['filteredParticles']}</b>
                            {temp_add_str}
                        </div>""", unsafe_allow_html = True)

                    st.subheader("Primary parameters")                    
                    st.markdown(f"""
                        <div class = 'text'>
                            Average diameter: <b>{np.mean(currentDiameter):.3f} nm</b> 
                        </div>""", unsafe_allow_html = True)
                    st.markdown(f"""
                        <div class = 'text'>
                            Standart deviation diameters: <b>{np.std(currentDiameter):.3f} nm</b> 
                        </div>""", unsafe_allow_html = True)


                    st.subheader("Secondary parameters")  
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
                    

                    st.subheader("Secondary parameters (norm)")                    
                    imageArea = np.prod(st.session_state['sizeImage'])
                    if st.session_state['scale'] is not None:
                        imageArea = imageArea * st.session_state['scale']**2
                       
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
                with db13.container(border = True, height = heightCol):
                    st.subheader("Heatmap of particle count")

                    currentBLOBs = st.session_state['BLOBs_filter']
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
                # END db13

            with st.expander("Some information charts (demo)", icon = ":material/data_thresholding:"):
                st.markdown(f"""
                    <p class = 'text'>
                        Visual representation of nanoparticle-based statistics in an image.
                        A detailed description is provided in the work on the first link below.
                    </p>""", unsafe_allow_html = True)

                currentBLOBs = st.session_state['BLOBs_filter']
                if st.session_state['scale'] is not None:         
                    currentBLOBs = currentBLOBs * st.session_state['scale']

                fullDist, minDist = NanoStat.euclideanDistance(currentBLOBs) 

                db21, db22, db23 = st.columns([1, 1, 1])
                st.markdown(f"""
                    <div class = 'about' style = "text-align: center;">
                        Any other information or chart
                    </div>""", unsafe_allow_html = True)

                # ?
                with db21.container(border = True, height = heightCol):              
                    x = np.arange(5, 100, 5)

                    emptySubareas = np.zeros_like(x, dtype = 'float')

                    for i, size in enumerate(x):
                        temp = NanoStat.uniformity(
                            st.session_state['BLOBs_filter'],
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

                # ?
                with db22.container(border = True, height = heightCol):                

                    fig = ff.create_distplot(
                        [minDist], [''], bin_size = 1, histnorm = 'probability',
                        colors = ['blue'], show_curve = False, show_rug = False
                    )

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Distance to the nearest nanoparticle", font = dict(size=27)),
                        xaxis_title_text = 'Distance to the nearest nanoparticle, nm',
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

                # ?
                with db23.container(border = True, height = heightCol):                
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
                

    ## TAB 4
    with tabGuide:
        # Guide 1
        st.subheader("Детектирование и фильтрация наночастиц")
        text_col, media_col = st.columns([1, 1])

        text_col.markdown(f"""
            <div>
                <p class = 'text'>Все дальнейшие шаги выполняются на владке "Automatic detection nanoparticles"!</p>
                <ul>
                    <li>
                        <p class = 'text'>
                            Шаг 1. Загрузка исходного СЭМ-изображения (кнопка "Browse file").
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Шаг 2. Детектирование наночастиц (кнопка "Nanoparticles detection" становится 
                            активной после загрузки изображения). Процесс детектирования занимает некоторое время,
                            в среднем до одной минуты.                      
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
                            (снять галочку "Use default settings"). ВАЖНО, подтверждение параметров детектирования 
                            осуществяется нажанием кнопки "Nanoparticles detection". Параметры фильтрации применяются
                            автоматически.
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)

        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)
                
        # Guide 2
        st.subheader("Взаимодейтсвие с результатами детектирования")
        text_col, media_col = st.columns([1, 1])

        text_col.markdown(f"""
            <div>
                <p class = 'text'>Указанный функционал доступен на владке "Automatic detection nanoparticles" после детектирования наночастиц!</p>
                <ul>
                    <li>
                        <p class = 'text'>
                            Результаты детектирования можно скачать в трёх вариантах:
                            (1) Найденные частицы на прозрачном фоне,
                            (2) Найденные частицы, наложенные на исходное изображение,
                            (3) В формате с указанием координат центра и радиуса каждой частицы.
                            Для этого нужно в выпадающем списке выбрать нужный вариант и нажать кнопку,
                            расположенную правее.
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Если на изображении присутсвует мерная шкала и её физический размер,
                            то масштаб определяется автоматически. Визуализировать оценённый масштаб
                            можно с помощью переключателя "Display scale".
                        </p>
                    </li>
                    <li>
                        <p class = 'text'>
                            Режим сравнения активируется с помощью переключателя "Comparison mode".
                            В этом режиме можно: (1) Скрыть\показать разметку на всём изображении нажатием ЛКМ,
                            (2) Скрыть\показать разметку в приближенной области нажатием ПКМ,
                            (3) Увеличить\уменьшить размер области приближения - колёсико мыши
                        </p>
                    </li>
                </ul>
            </div>""", unsafe_allow_html = True)
       
        media_col.markdown(f"""
            <div class = 'text' style = "text-align: center;">
                A video guide will be added here soon!
            </div>""", unsafe_allow_html = True)

        if st.button("Get exseption?"):
            raise Exception("Test exception!")

    ## How to cite
    st.markdown("""
        <div class = 'cite'> <b>How to cite</b>:
            <ul>
                <li> <p class = 'cite'>
                    Automated Recognition of Nanoparticles in Electron Microscopy Images of Nanoscale Palladium Catalysts.
                    <br> D. A. Boiko, V. V. Sulimova, M. Yu. Kurbakov [et al.] 
                    // Nanomaterials. – 2022. – Vol. 12, No. 21. – P. 3914. 
                    – DOI <a href=https://www.mdpi.com/2079-4991/12/21/3914>10.3390/nano12213914</a>
                </p> </li>
                <li> <p class = 'cite'>
                    Determining the Orderliness of Carbon Materials with Nanoparticle Imaging and Explainable Machine Learning. 
                    <br> M. Yu. Kurbakov, V. V. Sulimova, A. V. Kopylov [et al.] 
                    // Nanoscale. – 2024. – Vol. 16, No. 28. – P. 13663-13676. 
                    – DOI <a href=https://pubs.rsc.org/en/content/articlelanding/2024/nr/d4nr00952e>10.1039/d4nr00952e</a>.
                </p> </li>
                <li> <p class = 'cite'>
                    An article about calculating the mass of nanoparticles will be published soon, don't miss it!
                </p> </li>
            </ul>
        </div>""", unsafe_allow_html = True)
       

    st.markdown("""
        <div class = 'footer'>
            Laboratory of Cognitive Technologies and Simulating Systems, Tula State University © 2025 (E-mail: muwsik@mail.ru)
        </div>""", unsafe_allow_html = True)

except Exception as exc:
    dialog_exception(exc)