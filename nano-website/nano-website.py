# Run application
# streamlit run .\nano-website\nano-website.py

import streamlit as st

import io, csv, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from types import SimpleNamespace
import traceback

# charts
import plotly.express as px
import plotly.graph_objects as go

# content
import content.style as style
import content.tooltips as tooltips
import content.instructions as instruct

# utils
import utils.autoscale as autoscale
import utils.NanoStatistics as NanoStat
import utils.WebsiteBot as webBot
import utils.API2CVAT as API2CVAT
import utils.accuracy as accuracy

# dev utils
import utils.reworkExpApp as rEA

# UI
from streamlit_image_overlay import streamlit_image_overlay as overlay

    

### Function ###
    
def resetDetectionTab():
    st.session_state['Image'] = SimpleNamespace(
        upload = False,
        uploadedFile = None,
        source = None,
        preprocessed = None,
    )

    st.session_state['Particles'] = SimpleNamespace(
        detect = False,  
        data = None,
        settings = None,
        time = None,
    )

    st.session_state['Default'] = SimpleNamespace(
        imageType       = None,             # auto estimate
        d_brightness    = 5,
        d_diameter      = 2,                # type, not value 
        f_brightness    = 7,
        f_diameter      = (None, None),     # nm scale
        f_reliability   = 0.75,
    )

    # widgets
    st.session_state['Image.type'] = None
    st.session_state['Default.use'] = False
    st.session_state['Default.color'] = 'rgb(150, 150, 255)'
    st.session_state['Scale.display'] = False


def resetStatisticTab():    
    st.session_state['Statistic'] = SimpleNamespace(
        calculate = False,
        set = None,
        image = None,
        fileName = None,
    )

    st.session_state['distView'] = False
    st.session_state['normalize'] = False
    st.session_state['selected'] = False
    st.session_state['step'] = 0.5

    st.session_state['heatmap-step'] = 10


def resetStates(_dispToast = False):
    if _dispToast:
        st.toast('Default configuration loaded!')
    
    st.session_state['rerun'] = False
    
    st.session_state['Scale'] = None

    resetDetectionTab()
    resetStatisticTab()

    st.session_state["buffer"] = pd.DataFrame(columns = [
        "Image",
        "Scale unit",
        "Material type",
        "Number of particles",
        "Mean particle diameter",
        "Particle surface density, mg/m²",
        "Mean distance to neighbour",
        "Most probable distance to neighbour",
        "Distance threshold",
        "Fraction below distance threshold",
        "Clark-Evans index (R)",
    ])


def sessionState2str(closedKey = []):
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

        if st.session_state['Image'].uploadedFile is not None:
            dataException.update({                
                "image-data": st.session_state['Image'].uploadedFile.getvalue(),
                "image-type": st.session_state['Image'].uploadedFile.type
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
        if st.session_state['Image'].upload:
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
                "image-data": st.session_state['Image'].uploadedFile.getvalue(),
                "image-type": st.session_state['Image'].uploadedFile.type
            })

        result, _ = webBot.message2email(dataFeedback)
        
        if result:
            st.success("Feedback successful sent!")
        else:
            st.error("Error sending feedback. Please try again...")

    
@st.cache_data(show_spinner = False, max_entries = 5)
def detectScale(image):
    return autoscale.detectScale(image)


@st.cache_data(show_spinner = False, max_entries = 15)
def qualityEstimation(thres, gt_blobs, est_blobs):
    return accuracy.qualityEstimation(gt_blobs, est_blobs, thres) 


@st.cache_data(show_spinner = False, max_entries = 5)
def importTaskFromCVAT(taskCVAT):
    API2CVAT.importTaskFromCVAT(taskCVAT)


@st.cache_data(show_spinner = False, max_entries = 5)
def exportToCVAT(imageData, x, y, diameter):
    API2CVAT.exportToCVAT(imageData, x, y, diameter)


def addIoU2Overlays(overlays, iou):
    for overlay, value in zip(overlays, iou):
        overlay["tooltip"] += f"IoU: {value:.3f}\n"
    return overlays



### Main app ###
try:    
    # Initial loading of session states
    if 'rerun' not in st.session_state:
        resetStates()
    elif st.session_state['rerun']:
        resetStates(True)

    # Loading CSS styles
    st.set_page_config(page_title = "Web Nanoparticles", layout = "wide")
    style.loadStyles(st.session_state['Default.color'])
    
    ## Header
    instruct.Header()

    ## About
    instruct.About()

    ## Main content area
    tabDetect, tabStat, tabAccuracy, tabHelp = st.tabs([
        "Automatic detection",
        "Statistics dashboard",
        "Quality estimation",
        "Help"
    ])


    ## TAB 1
    with tabDetect:           
        st.subheader("Upload EM-image", anchor = False)  
        
        _tempUploadedFile = st.file_uploader("Choose an EM-image",
            label_visibility = 'collapsed',
            type = ["tif", "tiff", "png", "jpg", "jpeg" ]
        )

        if _tempUploadedFile is not None: 
            if (st.session_state['Image'].uploadedFile is None
                or st.session_state['Image'].uploadedFile.name != _tempUploadedFile.name
            ):                 
                resetDetectionTab() # Reset old values

                st.session_state['Image'].upload = True  
                st.session_state['Image'].uploadedFile = _tempUploadedFile  
                st.session_state['Image'].source = Image.open(_tempUploadedFile).convert("L").resize((1280, 960)) 
                # TO DO: resize fix?                
        else:
            resetDetectionTab()
        
        
        if (st.session_state['Image'].upload):
            colImage, colSetting = st.columns([6, 2])

            # Detection settings and results
            with colSetting: 
                # Preprocessing image
                with st.spinner("Preprocessing image...", show_time = True):                    
                    # defining the scale and related information
                    st.session_state['Scale'] = detectScale(st.session_state['Image'].source)
                           
                    st.session_state['Image'].preprocessed = st.session_state['Image'].source
                    if (st.session_state['Scale'].horizontalBound is not None): 
                        # Information bar crop
                        st.session_state['Image'].preprocessed = st.session_state['Image'].source.crop(
                            (
                                0,
                                0,
                                st.session_state['Image'].source.size[0], 
                                st.session_state['Scale'].horizontalBound
                            )
                        )                        

                    if (st.session_state['Image.type'] is None) or st.session_state['Default.use']:
                        _data = np.array(st.session_state['Image'].preprocessed, dtype = 'uint8').flatten()
                        counts, _ = np.histogram(_data, bins = np.arange(0, 255, 1))
                        counts = counts / np.sum(counts)

                        cumSum = 0
                        for i, j in enumerate(counts):
                            cumSum = cumSum + j
                            if (cumSum >= 0.5):
                                if (i <= 127):
                                    st.session_state['Image.type'] = 1     # SEM
                                else: st.session_state['Image.type'] = 2   # TEM
                                break
                                              
                st.toggle("Use default settings",
                    disabled = not st.session_state['Image'].upload,
                    key = 'Default.use',
                    help = tooltips.DefaultToggle
                )                         

                # Detection settings       
                with st.expander("Detection settings", icon = ":material/tune:",
                    expanded = not st.session_state['Particles'].detect,                    
                ):
                    st.pills("Type of microscope image",
                        required = True,
                        key = 'Image.type',
                        options = tooltips.Options.TypeMicroscope.keys(),
                        format_func = lambda option: tooltips.Options.TypeMicroscope[option],  
                        width = 'stretch',                     
                        disabled = st.session_state['Default.use'],
                        help = tooltips.TypeMicroscopePills
                    )                    
                    
                    if ('param-pre-1' not in st.session_state) or st.session_state['Default.use']:
                        st.session_state['param-pre-1'] = st.session_state['Default'].d_brightness
                        
                    st.slider("Minimal nanoparticle brightness",
                        key = 'param-pre-1',
                        disabled = st.session_state['Default.use'],
                        help = tooltips.Detection.Brightness
                    )

                    if ('param-pre-2' not in st.session_state) or st.session_state['Default.use']:
                            st.session_state['param-pre-2'] = st.session_state['Default'].d_diameter

                    st.selectbox("Hypothetical nanoparticles diameter",
                        key = 'param-pre-2',
                        options = tooltips.Options.NanopartSize.keys(),
                        format_func = lambda option: tooltips.Options.NanopartSize[option],
                        disabled = st.session_state['Default.use'],
                        help = tooltips.Detection.Diameter
                    )
        
                    pushDetectButton = st.button("Nanoparticles detection",
                        width = 'stretch',
                        disabled = not st.session_state['Image'].upload,
                        on_click = lambda: setattr(st.session_state["Particles"], "detect", True),
                    )
                    
                    tempWarningPlaceholder = st.empty()

                    if st.session_state['Particles'].detect:
                        if (st.session_state['Particles'].settings != [                        
                            st.session_state['Image.type'],
                            st.session_state['param-pre-1'],
                            st.session_state['param-pre-2'],
                        ]):
                            tempWarningPlaceholder.warning(
                                tooltips.Warnings.DetectSettings,
                                icon = ":material/warning:"
                            )
                
                # Detecting
                if pushDetectButton:
                    st.session_state['Particles'].detect = False
                    tempWarningPlaceholder.empty()
                    
                    timeStart = time.time()
                    with st.spinner("Nanoparticles detection...", show_time = True):                         
                        st.session_state['Particles'].detect = True                          
                        st.session_state['Particles'].settings = [
                            st.session_state['Image.type'],
                            st.session_state['param-pre-1'],
                            st.session_state['param-pre-2'],
                        ]   

                        st.session_state['Particles'].data = rEA.detectingParticles(
                            st.session_state['Image'].preprocessed,
                            st.session_state['Particles'].settings
                        )
                        st.session_state['Particles'].data.applyScale(st.session_state['Scale'].multiplier)   

                    st.session_state['Particles'].time = int(np.ceil(time.time() - timeStart))

                # Detection results
                if st.session_state['Particles'].detect:
                    instruct.DetectResult(
                        st.session_state['Particles'].data.detectedCount,
                        st.session_state['Particles'].time
                    )

                    # Warning about not correctly detection results 
                    if (st.session_state['Particles'].data.detectedCount < 1):            
                        st.warning(tooltips.Warnings.NoFoundNanos, icon = ":material/warning:")
                                    
                # Action with correctly detection results
                if (st.session_state['Particles'].detect and st.session_state['Particles'].data.detectedCount > 0):
                    # Filtration settings
                    with st.expander("Filtration settings", expanded = True, icon = ":material/filter_alt:"):
                        if ('param-filt-1' not in st.session_state) or st.session_state['Default.use']:
                            st.session_state['param-filt-1'] = st.session_state['Default'].f_brightness
                                                    
                        st.slider("Nanoparticle center brightness",
                            key = 'param-filt-1',
                            disabled = st.session_state['Default.use'],
                            help = tooltips.Filtration.Brightness
                        )

                        # Settings slider with diameters    
                        _diameters = st.session_state['Particles'].data.get('diameter', False)                        
                        min_d = np.min(_diameters)
                        max_d = np.percentile(_diameters, 97)

                        slider_min = np.floor(min_d / 10) * 10
                        slider_max = np.ceil(max_d / 10) * 10

                        if slider_max <= slider_min:
                            slider_max = slider_max + 10

                        if ('param-filt-2' not in st.session_state) or st.session_state['Default.use']:
                            st.session_state['param-filt-2'] = (min_d, max_d)

                        st.slider(f"Nanoparticles diameter, {st.session_state['Scale'].unit}",
                            key = 'param-filt-2',                         
                            min_value = slider_min,
                            step = 0.1,
                            max_value = slider_max,
                            format = "%0.1f",
                            disabled = st.session_state['Default.use'],
                            help = tooltips.Filtration.Diameter +
                                f"""**{np.sum(_diameters > max_d)}** 
                                    particles exceed the 97th with a diameter greater 
                                    than {max_d:.1f} {st.session_state['Scale'].unit}"""                             
                        )


                        if ('param-filt-3' not in st.session_state) or st.session_state['Default.use']:
                            st.session_state['param-filt-3'] = st.session_state['Default'].f_reliability

                        st.slider("Nanoparticle reliability",
                            key = 'param-filt-3',
                            min_value = 0.0,
                            step = 0.01,
                            max_value = 1.0,
                            disabled = st.session_state['Default.use'],
                            help = tooltips.Filtration.Reliability
                        )
                        
                    # Filtering                    
                    st.session_state['Particles'].data.setfilter( 
                        c0 = (st.session_state['param-filt-1'], None),
                        diameter = (
                            st.session_state['param-filt-2'][0],
                            None if np.isclose(slider_max, st.session_state['param-filt-2'][1]) 
                                else st.session_state['param-filt-2'][1]
                        ),
                        approxError = (None, 1 - st.session_state['param-filt-3'])
                    )
                    
                    # Info about filtered nanoparticles
                    instruct.FiltrationResult(st.session_state['Particles'].data.count)

                    if (st.session_state['Particles'].data.count < 1):
                        st.warning(tooltips.Warnings.FiltrSettings, icon = ":material/warning:")                                      
                                        
                    with st.expander("Visualization and saving results", expanded = False,
                        icon = ":material/display_settings:"
                    ):
                        # Displaying the scale
                        st.toggle("Estimated scale",
                            key = 'Scale.display', 
                            disabled = st.session_state['Scale'].info is None,
                            help = tooltips.Visualization.Scale +
                                (tooltips.Warnings.OutScale if st.session_state['Scale'].info is None else "")
                        )
                            
                        # Saving
                        #TODO: use container horizontal
                        selectboxCol, buttonCol = st.columns([6,1], vertical_alignment = 'bottom')

                        _selectionSave = selectboxCol.selectbox(
                            "What results should be saved?",
                            index = 3,
                            placeholder = "Select options...",
                            options = tooltips.Options.Saving.keys(),
                            format_func = lambda option: tooltips.Options.Saving[option]
                        )

                        fileResult = io.BytesIO()
                        fileResultName = Path(st.session_state['Image'].uploadedFile.name).stem
                        _buttonDownloadDisabled = False

                        match _selectionSave:
                            case 0:
                                _temp = st.session_state['Particles'].data.paint(
                                    Image.new(mode = "RGBA", size = st.session_state['Image'].preprocessed.size),
                                    st.session_state['Default.color']
                                )
                                _temp.save(fileResult, format = 'png')
                                fileResultName += f"_particles.tif"
                            case 1:
                                _temp = st.session_state['Particles'].data.paint(
                                    st.session_state['Image'].source.convert("RGB"),
                                    st.session_state['Default.color']
                                )
                                _temp.save(fileResult, format = 'png')
                                fileResultName += f"_particls+image.tif"

                            case 2:
                                fileResult = io.StringIO()
                                _tempWriter = csv.writer(fileResult, delimiter = ';')
                                _tempWriter.writerow([f"Scale: {st.session_state['Scale'].multiplier:.3} ({st.session_state['Scale'].unit}/px)"])
                                _tempWriter.writerow(['coord x, px', 'coord y, px', 'diameter, px'])
                                _tempWriter.writerows(zip(
                                    st.session_state["Particles"].data.get('x_px'),
                                    st.session_state["Particles"].data.get('y_px'),
                                    st.session_state["Particles"].data.get('diameter_px'),
                                ))
                                fileResultName += f"_parameters.csv"
                            case 3:
                                tempWidth, tempHeight = st.session_state["Image"].source.size
                                imageData = {
                                    'name': Path(st.session_state['Image'].uploadedFile.name).stem,
                                    'width': tempWidth,
                                    'height': tempHeight,
                                    'buffer': st.session_state['Image'].uploadedFile.getvalue()
                                }
                                fileResult = exportToCVAT(
                                    imageData, 
                                    st.session_state["Particles"].data.get('x_px'),
                                    st.session_state["Particles"].data.get('y_px'),
                                    st.session_state["Particles"].data.get('diameter_px'),
                                )
                                fileResultName += f"_{time.strftime('%Y-%m-%d-%H-%M-%S')}.zip"
                            case _:
                                _buttonDownloadDisabled = True

                        buttonCol.download_button(
                            label = "",
                            icon = ":material/download:",
                            data = fileResult.getvalue(),
                            file_name = fileResultName,
                            disabled = _buttonDownloadDisabled,
                            help = tooltips.Visualization.Download
                        )
         
            
            # Display image 
            with colImage:
                _overlays = []
                if st.session_state['Scale.display']:
                    y, x, length, text = st.session_state['Scale'].info
                    x += 2 
                    tick = 5
                    _overlays += [
                        {
                        "id": "scale",
                        "type": "path",
                        "data": {
                            "d": (
                                f"M {x} {y - tick} "
                                f"L {x} {y + tick} "
                                f"M {x} {y} "
                                f"L {x + length} {y} "
                                f"M {x + length} {y - tick} "
                                f"L {x + length} {y + tick}"
                            ),
                        },
                        "tooltip": (
                            f"Estimated scale: {st.session_state["Scale"].multiplier:.4f} nm/px\n"
                            f"Recognition text: {text}\n"
                        ),
                    }                        
                    ]

                if st.session_state["Particles"].data is not None:
                    _overlays += st.session_state["Particles"].data.toOverlays(
                            unit = st.session_state["Scale"].unit
                        )

                overlay(
                    image = st.session_state["Image"].source,
                    overlays = _overlays,
                    styles = {
                        "viewport": {
                            "height": "85vh",                 
                            "border": "1px sold #fff",
                            "border-radius": "5px",
                        },
                        "tooltip": {
                            "background-color": "black",
                            "color": "white",
                            "border-radius": "5px",
                            "padding": "10px",
                            "font-size": "15px",
                            "white-space": "pre-line"
                        },
                        "circle": {
                            "default": {
                                "stroke": st.session_state['Default.color'],
                                "strokeWidth": 1,
                            }
                        },
                        "path": {
                            "default": {
                                "stroke": st.session_state['Default.color'],
                                "strokeWidth": 2,
                            }
                        }
                    },
                    key = "main-imageViewer"                    
                )


    ## TAB 2 
    with tabStat:    
        heightCol = 550
        marginChart = dict(l=10, r=10, t=10, b=5)
              
        with st.expander("Global dashboard settings",
            expanded = True,
            icon = ":material/rule_settings:"
        ):
            _selectionUseNano = st.selectbox(
                "Which nanoparticles to use?",
                #key = "Statistic.use",
                index = 2,
                options = tooltips.Options.NanoStatistic.keys(),
                format_func = lambda option: tooltips.Options.NanoStatistic[option],
                help = tooltips.NanopartSelectbox
            ) 
                
            st.session_state['Statistic'].calculate = False
            match _selectionUseNano:
                case 0:
                    if (not st.session_state['Particles'].detect):
                        st.warning(tooltips.Warnings.NoResults, icon = ":material/warning:")

                    elif (st.session_state['Particles'].data.count < 10):
                        st.warning(tooltips.Warnings.SmallResults, icon = ":material/warning:")

                    else:                        
                        st.session_state['Statistic'].calculate = True

                        st.session_state['Statistic'].data = st.session_state['Particles'].data
                        st.session_state['Statistic'].fileName = Path(st.session_state['Image'].uploadedFile.name).stem
                        st.session_state['Statistic'].image = st.session_state['Image'].source
                        st.session_state['sizeImage'] = st.session_state['Image'].preprocessed.size
                case 1:
                    instruct.LabelUploderFileCVAT()
                    uploadedFileCVAT = st.file_uploader("Uploder CVAT file",
                        type = ["zip"],
                        label_visibility = 'collapsed'
                    )

                    if uploadedFileCVAT is not None:
                        try:
                            _blobs, _fileName, _imageCVAT = API2CVAT.ImportTaskFromCVAT(uploadedFileCVAT)
                        except:
                            st.warning(tooltips.Warnings.BadFileFormat, icon = ":material/warning:") 
                            _blobs = []

                        if len(_blobs) == 0:
                            st.warning(tooltips.Warnings.NoResultsCVAT, icon = ":material/warning:")
                        else:            
                            st.session_state['Statistic'].calculate = True                 
                            st.session_state['Statistic'].image = Image.open(_imageCVAT).convert("L").resize((1280, 960)) #TODO: fix resize?
                            st.session_state['Statistic'].fileName = Path(_fileName).stem

                            #TODO: The global scale is used to automatically detect and download the CVAT.
                            #  It's working now because the first layout is always executed and the image scale
                            #  is always used there. and here it works because the scale is used from a backup.
                            #  It is better to reduce these scales.
                            st.session_state['Scale'] = detectScale(st.session_state['Statistic'].image)

                            #TODO: remove  st.session_state['sizeImage']
                            st.session_state['sizeImage'] = list(st.session_state['Statistic'].image.size)
                            if (st.session_state['Scale'].horizontalBound is not None): 
                                st.session_state['sizeImage'][1] = st.session_state['Scale'].horizontalBound
                                
                            st.session_state['Statistic'].data = rEA.ParticleSet(
                                _blobs,
                                st.session_state['Scale'].multiplier
                            )
                case _:
                    pass

        if (not st.session_state['Statistic'].calculate):
            resetStatisticTab()
        else:
            with st.expander("Particle parameters", expanded = True,
                icon = ":material/app_registration:"
            ):
                instruct.AboutSectionParticleParams()

                statistic_d = st.session_state["Statistic"].data.get('diameter')

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
                    start = np.floor(statistic_d.min()) - step
                    end = np.ceil(statistic_d.max()) + step

                    counts, bins = np.histogram(statistic_d, bins = np.arange(start, end, step, dtype = float))
                                        
                    name_x = f"Diameters, {st.session_state["Scale"].unit}"
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

                    fig = go.Figure().add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = bar_y,
                        customdata = customDataChart,
                        showlegend = False,
                        hovertemplate = (
                            f"Diameter: [%{{customdata[0]:.1f}}, %{{customdata[1]:.1f}}) {st.session_state["Scale"].unit}<br>"
                            "Particls: " + hover_y +
                            "<extra></extra>"
                        ),
                        marker = dict(
                            color = st.session_state['Default.color'],
                            line_color = 'blue',
                            line_width = 1  
                        ),
                    ))

                    if st.session_state['distView']:
                        mu = np.mean(statistic_d)
                        sigma = np.std(statistic_d)

                        dist_x = np.arange(start, end, step * 0.1, dtype = float)
                        dist_y = np.exp(-1/2 * ((dist_x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

                        fig.add_trace(go.Scatter(
                            x = dist_x, 
                            y = dist_y * step * (100 if st.session_state['normalize'] else len(statistic_d)),
                            mode = 'lines',
                            hoverinfo = 'skip',
                            showlegend = False,
                            line = dict(color = 'rgba(50, 50, 255, 0.75)'),                            
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x = [None], 
                            y = [None],
                            mode = 'lines',
                            line = dict(width = 0),    
                            showlegend = True,
                            name = f"Particles: {st.session_state['Statistic'].data.count}<br>"
                                + f"Avg. diameter: {mu:0.1f} {st.session_state["Scale"].unit}<br>"
                                + f"Std. dev. diameter: {sigma:0.1f} {st.session_state["Scale"].unit}" 
                        )) 
                        
                    fig.update_layout(
                        margin = marginChart,
                        xaxis = dict(
                            title = name_x,
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
                        ),
                        yaxis_title_text = name_y,                        
                        bargap = 0,
                        legend = dict(
                            x = 1,
                            y = 1,
                            xanchor = 'right',
                            yanchor = 'top',
                            bgcolor = 'rgba(0,0,0,0)'
                        )
                    )

                    st.plotly_chart(
                        fig,
                        width = 'stretch',
                        selection_mode = 'points'
                    )

                                           
                    file = io.StringIO()
                    csv.writer(file, delimiter = ';').writerow([name_x, name_y])
                    csv.writer(file, delimiter = ';').writerows(dataChart)
                                        
                    buttonDataChartPlaceholder.download_button(
                        label = "Download data chart *.csv",
                        data = file.getvalue(),
                        file_name = f"{st.session_state['Statistic'].fileName}-dist-diameters.csv",
                        width = 'stretch',
                        help = tooltips.Distribution.Download
                    )
                # END db11

                # Nanoparticle parameters
                with db12.container(border = True, height = heightCol): 
                    with st.popover("Nanoparticle parameters", width = 'stretch'):
                        #TODO: _selectedMaterial added in st.session_state
                        _selectedMaterial = st.pills(
                            "Particles material",
                            default = 0,
                            required = True,
                            width = 400,
                            options = tooltips.Options.MaterialName.keys(),
                            format_func = lambda option: tooltips.Options.MaterialName[option],
                        )

                        materialName = tooltips.Options.MaterialName[_selectedMaterial]

                        if _selectedMaterial == 4:
                            materialDensity = st.number_input(
                                "Particles material density on ng/nm³",
                                min_value = 0.0,
                                step = 1.0e-11,
                                value = 1.0e-10,
                                format = "%0.2e",
                                key = "user-density"
                            )
                        else:
                            materialDensity = tooltips.Options.MaterialDensity[_selectedMaterial]

                        instruct.MaterialDensity(None, materialDensity)
                    
                    # Additional info                                     
                    instruct.EstimatedScale(st.session_state["Scale"]) # TODO input scale
                    
                    if _selectedMaterial == 4: # User material
                        instruct.MaterialDensity(materialName, materialDensity)
                    else:
                        instruct.MaterialDensity(materialName, None)
                                        
                    instruct.Quantity(st.session_state['Statistic'].data.count)

                    instruct.AboutPrimaryParameters()

                    instruct.MeanDiameter(
                        np.mean(statistic_d),
                        st.session_state['Scale'].unit
                    )

                    instruct.StdDiameter(
                        np.std(statistic_d),
                        st.session_state['Scale'].unit
                    )

                    instruct.AboutSecondaryParameters()

                    _area = np.sum(st.session_state['Statistic'].data.get('area'))
                    instruct.Area(
                        _area,
                        st.session_state['Scale'].unit
                    )

                    _volume = np.sum(st.session_state['Statistic'].data.get('volume'))
                    instruct.Volume(
                        _volume,
                        st.session_state['Scale'].unit
                    )     

                    _mass = _volume * materialDensity
                    instruct.Mass(None if st.session_state['Scale'].unit == 'px' else _mass)

                    _imageArea = np.prod(st.session_state['Scale'].apply(st.session_state['sizeImage']))
                    instruct.AboutNormSecondaryParameters(
                        _imageArea,
                        st.session_state['Scale'].unit
                    )

                    instruct.NormArea(_area / _imageArea * 100)
                
                    instruct.NormMass(
                        None if st.session_state['Scale'].unit == 'px' else (_mass / _imageArea)
                    )                         
                # END db12

                # Visualization particles
                with db13.container(border = True, height = heightCol):
                    with st.popover("Visualization particles", width = 'stretch'):                    
                        _tempSelectionChart = st.pills(
                            "Type visualization",
                            default = 1,
                            required = True,
                            options = tooltips.Options.TypeChart.keys(),
                            format_func = lambda option: tooltips.Options.TypeChart[option],
                            label_visibility = 'collapsed'
                        )

                        if _tempSelectionChart == 0:
                            st.slider("Step of splitting image for heatmap",
                                min_value = 10,                           
                                step = 1,
                                max_value = 25,
                                key = 'heatmap-step',
                                help = tooltips.NoneInfo
                            )

                    match _tempSelectionChart:
                        case 0: 
                            _heightBlocks = int(np.ceil(st.session_state['sizeImage'][1] / st.session_state['heatmap-step']))
                            _widthBlocks = int(np.ceil(st.session_state['sizeImage'][0] / st.session_state['heatmap-step']))
    
                            uniformityMap = NanoStat.uniformity(
                                st.session_state['Statistic'].data.get('x_px'),
                                st.session_state['Statistic'].data.get('y_px'),
                                (_heightBlocks, _widthBlocks),
                                st.session_state['heatmap-step'],
                            )

                            fig = px.imshow(uniformityMap, aspect = "equal")

                            fig.update_traces(
                                hovertemplate = "Particles in subarea: %{z:.0f}<extra></extra>"
                            )

                            fig.update_layout(
                                margin = marginChart,
                                xaxis_title_text = 
                                    f'Width image, {st.session_state['heatmap-step']}*px',
                                yaxis_title_text = 
                                    f'Height image, {st.session_state['heatmap-step']}*px',
                                coloraxis_colorbar = dict(
                                    title = "Particle count",
                                    orientation = "h",
                                    y = -0.2,
                                ),
                                showlegend = False
                            )

                            st.plotly_chart(fig, width = 'stretch')
                        case 1: 
                            st.image(
                                st.session_state['Statistic'].data.paint(
                                    st.session_state['Statistic'].image.convert("RGB"),
                                    st.session_state['Default.color']
                                ),
                                width = 'stretch'
                            )
                # END db13

            with st.expander("Nanoparticle spatial distribution", expanded = True,
                icon = ":material/data_thresholding:"
            ):
                instruct.AboutSectionSpatialDistribution()

                fullDist, minDist = NanoStat.euclideanDistance(
                    st.session_state['Statistic'].data.get('x_px'),
                    st.session_state['Statistic'].data.get('y_px'),
                ) 

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
                        _heightBlocks = int(np.ceil(st.session_state['sizeImage'][1] / size))
                        _widthBlocks = int(np.ceil(st.session_state['sizeImage'][0] / size))

                        _map = NanoStat.uniformity(
                            st.session_state['Statistic'].data.get('x_px'),
                            st.session_state['Statistic'].data.get('y_px'),
                            (_heightBlocks, _widthBlocks),
                            size,
                        )
                        emptyCount[i] = np.sum(_map == 0)
                        totalCount[i] = _map.size
                        emptySubareas[i] = emptyCount[i] / totalCount[i]                    
                    
                    fig = go.Figure().add_trace(go.Bar(
                        x = st.session_state['Scale'].apply(x),
                        y = emptySubareas,                                 
                        hovertemplate = (
                            f"Size: %{{x:.2}} {st.session_state['Scale'].unit}"
                            f"<br>Empty: %{{y:.2}}<extra></extra>"
                        ),
                        marker_color = st.session_state['Default.color'],
                        marker_line_color = 'blue',
                        marker_line_width = 1,
                    ))

                    fig.update_layout(
                        margin = marginChart,
                        xaxis = dict(
                            title = "Size of square subareas, " + st.session_state['Scale'].unit,
                            showgrid = True,
                        ),
                        yaxis = dict(
                            title = "Empty subareas fraction",
                        ),
                        showlegend = False,
                        bargap = 0
                    )

                    st.plotly_chart(fig, width = 'stretch',)

                    # Saving db21 
                    db21_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Block size ({st.session_state['Scale'].unit})": st.session_state['Scale'].apply(x),
                            "Empty subareas": emptyCount,
                            "Total subareas": totalCount,
                            "Empty fraction": emptySubareas,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['Statistic'].fileName}_empty-subareas.csv",
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

                    bins = np.append(np.arange(0, 52, 2), np.inf)
                    counts, _ = np.histogram(minDist, bins = bins)                      
                    distanceNearest = counts / np.sum(counts)

                    fig = go.Figure().add_trace(go.Bar(
                        x = 0.5 * (bins[:-2] + bins[1:-1]),
                        y = distanceNearest[:-1],
                        showlegend = False,
                        hovertemplate = (
                            f"Distanse: %{{x:.2}} {st.session_state['Scale'].unit}"
                            f"<br>Fraction: %{{y:.2}}<extra></extra>"
                        ),
                        marker = dict(
                            color = st.session_state['Default.color'],
                            line_color = 'blue',
                            line_width = 1, 
                        )
                    ))

                    fig.add_trace(go.Bar(
                        x = [51],
                        y = [distanceNearest[-1]],
                        hovertemplate = (
                            f"Distanse >50 {st.session_state['Scale'].unit}"
                            f"<br>Fraction: %{{y:.2}}<extra></extra>"
                        ),
                        showlegend = False,
                        marker = dict(
                            color = st.session_state['Default.color'],
                            line_color = 'blue',
                            line_width = 1, 
                        )
                    ))
                        
                    fig.update_layout(
                        margin = marginChart,                        
                        bargap = 0,
                        xaxis = dict(
                            title =  f'Distance to nearest nanoparticle, {st.session_state["Scale"].unit}',
                            showgrid = True,
                        ),
                        yaxis_title_text = 'Particle fraction',
                        showlegend = False
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                                        
                    # Saving db22
                    db22_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Distance ({st.session_state["Scale"].unit})": minDist,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['Statistic'].fileName}_distance-nearest-nanoparticle.csv",
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
                    x = st.session_state["Scale"].apply(np.arange(5, 105, 1))

                    averageDensity = NanoStat.averageDensityInNeighborhood(x, fullDist)
                    numberLess = np.rint(averageDensity * np.pi * x**2 * len(fullDist)).astype(int)
                                                            
                    fig = go.Figure().add_trace(go.Bar(
                        x = x,
                        y = averageDensity,                        
                        hovertemplate = (
                            f"Neighborhood radius: %{{x:.2}} {st.session_state["Scale"].unit}"
                            f"<br>Particles/{st.session_state["Scale"].unit}²: %{{y:.1e}}<extra></extra>"
                        ),
                        marker = dict(
                            color = st.session_state['Default.color'],
                            line_color = 'blue',
                            line_width = 0.5
                        ),
                    ))

                    fig.update_layout(
                        margin = marginChart,
                        xaxis = dict(
                            title = f'Neighborhood radius, {st.session_state["Scale"].unit}',                                
                            showgrid = True,
                        ),
                        yaxis_title_text = 
                            f'Nanoparticles per unit area, particles/{st.session_state["Scale"].unit}²',
                        showlegend = False,
                        bargap = 0
                    )

                    st.plotly_chart(fig, width = 'stretch')
                    
                    # Saving db23
                    db23_buttonPlaceholder.download_button(
                        label = "Download raw data chart *.csv",
                        data = pd.DataFrame({
                            f"Neighborhood radius ({st.session_state["Scale"].unit})": x,
                            "Number of particles": len(fullDist),
                            "Number of neighbors": numberLess,
                            f"Average density (particles/{st.session_state["Scale"].unit}^2)": averageDensity,
                        }).to_csv(index = False).encode("utf-8"),
                        file_name = f"{st.session_state['Statistic'].fileName}_nanoparticles-unit-area.csv",
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

                    x = st.session_state["Scale"].apply(np.arange(10, 105, 1))
                    localArea = NanoStat.localAreaFraction(x, fullDist, statistic_d)
                    
                    fig = go.Figure().add_trace(go.Bar(
                        x = x,
                        y = localArea,                        
                        hovertemplate = (
                            f"Area size: %{{x:.2}} {st.session_state["Scale"].unit}"
                            f"<br>Coverage: %{{y:.1%}}<extra></extra>"
                        ),
                        marker = dict( 
                            color = st.session_state['Default.color'],
                            line_color = 'blue',
                            line_width = 0.5
                        ),
                    ))

                    fig.update_layout(
                        margin = marginChart,
                        xaxis = dict(
                            title = f'Neighbors area size, {st.session_state["Scale"].unit}',
                            showgrid = True,
                        ),
                        yaxis_title_text = 'Neighbors coverage, %',
                        showlegend = False,
                        bargap = 0
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                # END db31

                # Average number of neighbors
                with db32.container(border = True, height = heightCol):  
                    with st.popover("Average number of neighbors", width = 'stretch'):
                        pass               
                    
                    x = st.session_state["Scale"].apply(np.arange(10, 105, 1))
                    averageNeighborhoods = NanoStat.averageNeighborhoods(x, fullDist)
                    
                    fig = go.Figure().add_trace(go.Bar(
                        x = x,
                        y = averageNeighborhoods,                        
                        hovertemplate = (
                            f"Size: %{{x:.2}} {st.session_state["Scale"].unit}"
                            f"<br>Neighbors: %{{y:.3}}<extra></extra>"
                        ),
                        marker = dict(
                           color = st.session_state['Default.color'],
                            line_color = 'blue',                        
                            line_width = 0.5,
                        ),
                    ))

                    fig.update_layout(
                        margin = marginChart,
                        xaxis = dict(
                            title = f'Nanoparticle neighborhood size, {st.session_state["Scale"].unit}',                            
                            showgrid = True,
                        ),
                        yaxis_title_text = 'Average number of neighbors',
                        showlegend = False,
                        bargap = 0
                    )

                    st.plotly_chart(fig, width = 'stretch',)
                # END db32
            
                # Statistics aggregator
                with db33.container(border = True, height = heightCol):
                    st.subheader("Aggregate statistics", width = 'stretch') 

                    # check the presence of this image in table
                    if st.session_state['Statistic'].fileName != st.session_state.get("prevImage"):
                        st.session_state["prevImage"] = st.session_state['Statistic'].fileName
                        if not st.session_state["buffer"]["Image"].eq(st.session_state['Statistic'].fileName).any():                                              
                            newRow = {
                                "Image": st.session_state["Statistic"].fileName,
                                "Material type": materialName,
                                **NanoStat.aggregateStatistics(
                                    statistic_d,
                                    minDist,
                                    materialDensity = materialDensity,
                                    imageArea = np.prod(
                                        st.session_state["Scale"].apply(st.session_state["sizeImage"])
                                    ),
                                    scaleUnit = st.session_state["Scale"].unit,
                                ),
                            }
                        
                            st.session_state["buffer"] = pd.concat(
                                [ st.session_state["buffer"], pd.DataFrame([newRow]) ],
                                ignore_index = True
                            )
                            
                            st.toast(f"**Added** statistics for file '{st.session_state['Statistic'].fileName}'")
                        else:
                            st.toast(f"File '{st.session_state['Statistic'].fileName}' is **already in** the aggregated statistics")
                    
                    buffer = st.session_state["buffer"].copy()
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

                    if st.session_state['Scale'].unit == 'px':
                        st.warning(tooltips.Warnings.IncompleteStats, icon = ":material/warning:")

                    deleteMask = editedBuffer["Delete"].astype(bool)

                    if deleteMask.any():
                        st.session_state["buffer"] = (
                            editedBuffer.loc[~deleteMask]
                                .drop(columns = "Delete")
                                .reset_index(drop = True)
                        )
                        st.rerun()  # !!!
                # END db33


    ## TAB 3    
    with tabAccuracy:
        instruct.AboutSectionQuality()

        uploadedFileGT = st.file_uploader("Expert markup file", type = ["zip"],
            help = tooltips.ExpertFileUploader
        )
                    
        if uploadedFileGT is not None:
            try:
                gt_blobs, _, _ = importTaskFromCVAT(uploadedFileGT) 
            except:
                st.warning(tooltips.Warnings.BadFileFormat, icon = ":material/warning:")
                gt_blobs = []

            if len(gt_blobs) == 0:
                st.warning(tooltips.Warnings.NoResultsCVAT, icon = ":material/warning:")
            elif st.session_state['Particles'].data is None:                
                st.warning(tooltips.Warnings.NoResults, icon = ":material/warning:")
            else:                
                st.session_state['sizeImage'] = st.session_state['Image'].preprocessed.size
               
                est_blobs = np.column_stack([
                    st.session_state['Particles'].data.get('x_px'),
                    st.session_state['Particles'].data.get('y_px'),
                    st.session_state['Particles'].data.get('diameter_px'),
                ])

                l, r = st.columns([3, 7])

                with l:                    
                    _thres = st.slider("Jacquard measure threshold",
                        min_value = 0.05,
                        step = 0.01,
                        max_value = 0.95,
                        value = 0.25,
                        help = tooltips.NoneInfo                      
                    )

                    #TODO: added IoU for all particles
                    FN, FP, TP, TD, TD_IoU = qualityEstimation(_thres, gt_blobs, est_blobs)

                    instruct.Quality(len(FN), len(FP), len(TP))
                    instruct.LegendChartQuality()

                with r:
                    overlay(
                        key = 'accuracy',
                        image = st.session_state['Image'].source,
                        overlays = 
                            addIoU2Overlays(
                                rEA.ParticleSet(TD).toOverlays(classes = 'detect', info = "None"),
                                TD_IoU
                            )
                            +                     
                            rEA.ParticleSet(TP).toOverlays(classes = 'TP', info = "None")
                            +                        
                            rEA.ParticleSet(FP).toOverlays(classes = 'FP', info = "None")
                            +                        
                            rEA.ParticleSet(FN).toOverlays(classes = 'FN', info = "None")
                            ,
                        styles = {     
                            "viewport": {
                            },
                            "tooltip": {
                                "background-color": "black",
                                "color": "white",
                                "white-space": "pre-line"
                            },
                            "circle": {
                                "class": {
                                    "detect": {
                                        "stroke": "blue",
                                        "stroke-width": 1,
                                    },                            
                                    "TP": {
                                        "stroke": "green",
                                        "stroke-width": 1,
                                    },                            
                                    "FN": {
                                        "stroke": "red",
                                        "stroke-width": 1,
                                    },                            
                                    "FP": {
                                        "stroke": "yellow",
                                        "stroke-width": 1,
                                    },
                                }
                            }                        
                        }
                    )


    ## TAB 4
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