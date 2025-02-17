import streamlit as st

import io, csv
import numpy as np
from skimage import morphology
from skimage.filters import median
from streamlit_image_comparison import image_comparison
from PIL import Image, ImageDraw
   
import plotly.express as px
import plotly.figure_factory as ff

import style, tools, autoscale

# Run
# streamlit run .\nano-website\nano_website.py --server.enableXsrfProtection false

help_str = "be added soon"


def load_default_settings():
    st.session_state['imageUpload'] = False
    st.session_state['uploadedImage'] = None

    st.session_state['settingDefault'] = True

    st.session_state['detected'] = False
    st.session_state['BLOBs'] = None
    st.session_state['imageBLOBs'] = None
    st.session_state['sizeImage'] = None

    st.session_state['comparison'] = False
    
    st.session_state['scale'] = None

    st.session_state['chartRange'] = ('min','max')
    st.session_state['distView'] = False
    st.session_state['recalculation'] = True

if 'imageUpload' not in st.session_state:
    load_default_settings()


# Header
style.set_style()
st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)

st.markdown("""<div class = 'about'>
                    Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
                    <br>It will help you to detect palladium nanoparticles in the image and calculate their mass.
               </div>""", unsafe_allow_html = True)

st.markdown("""<div class = 'about'>
                    Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
               </div>""", unsafe_allow_html = True)


# Main content area
tabDetect, tabInfo = st.tabs(["Detection nanoparticls", "Statistics dashboard"])


# TAB 1
with tabDetect:
    left, rigth = st.columns([8, 3])

    # Viewing images
    with left:
        st.header("Upload SEM image")
        uploadedImage = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "tif"])
    
        imagePlaceholder = st.empty()

        if uploadedImage is None:        
             load_default_settings()
        else:
            st.session_state['imageUpload'] = True
            crsImage = Image.open(uploadedImage)
            grayImage = np.array(crsImage.convert('L'), dtype = 'uint8')
            
            if (not np.array_equal(st.session_state['uploadedImage'], grayImage)):
                st.session_state['uploadedImage'] = grayImage
                st.session_state['detected'] = False   
                
                #
                tempScale = autoscale.estimateScale(grayImage)
                if tempScale is not None:
                    st.session_state['scale'] = tempScale

            if (not st.session_state['detected']):
                imagePlaceholder.image(crsImage, use_column_width = True, caption = "Uploaded image")
            elif (not st.session_state['comparison']):
                imagePlaceholder.image(st.session_state['imageBLOBs'], use_column_width = True, caption = "Detected nanoparticles")        
            else:
                st.markdown(
                    f"""
                        <style>
                        iframe {{
                            width: inherit;
                            height: 1150px;
                        }}
                        </style>
                    """, unsafe_allow_html = True)

                image_comparison(
                   img1 = crsImage,
                   img2 = st.session_state['imageBLOBs'],
                   label1 = "Initial SEM image",
                   label2 = "Detected nanoparticles",
                   in_memory = True)
    # END left side

    # Detection settings and results
    with rigth:
        st.header("Settings")

        st.checkbox("Use default settings?",
            disabled = not st.session_state['imageUpload'],
            key = 'settingDefault',
            help = "You need to upload an SEM image")
    
        # Preprocessing settings
        with st.container(border = True):
            methodPrep = st.selectbox("Image preprocessing method",
                    ("None", "Top-hat  &  threshold filtering", "New!"),
                    index = 1 if st.session_state['settingDefault'] else 0,
                    disabled = st.session_state['settingDefault'],
                    help = help_str)

            if methodPrep == "Top-hat  &  threshold filtering":
                thrPrepCoef = st.text_input("Pretreatment filtering coefficient",
                    value = "0.2" if st.session_state['settingDefault'] else "0.3",
                    disabled = st.session_state['settingDefault'],
                    help = help_str)

            dispCurrentImage = st.checkbox("Display image after preprocessing?",
                value = False if st.session_state['settingDefault'] else True,
                disabled = st.session_state['settingDefault'],
                help = help_str)
    
        # Nanoparticle detection settings
        with st.container(border = True):        
            methodDetect = st.selectbox("Nanoparticle detection method",
                    ("Exponential approximation", "New!"),
                    index = 0 if st.session_state['settingDefault'] else 0,
                    disabled = st.session_state['settingDefault'],
                    help = help_str)

            if methodDetect == "Exponential approximation":
                thresCoefOld = st.text_input("Nanoparticle brightness filtering threshold",
                    value = "0.5" if st.session_state['settingDefault'] else "0.4",
                    disabled = st.session_state['settingDefault'],
                    help = help_str)

                fsize = st.selectbox("Size of the approximation window (in pixels)",
                        (5, 7, 9, 11, 13),
                        index = 1 if st.session_state['settingDefault'] else 0,
                        disabled = st.session_state['settingDefault'],
                        help = help_str)


        detectButtonPush = st.button("Nanoparticles detection",
            use_container_width = True,
            disabled = not st.session_state['imageUpload'],
            help = "You need to upload an SEM image"
        )
    
        # Detecting
        if detectButtonPush:
            currentImage = np.copy(grayImage) 
         
            lowerBound = autoscale.findBorder(grayImage)        
            if (lowerBound is not None):
                currentImage = currentImage[:lowerBound, :]
            
            st.session_state['sizeImage'] = currentImage.shape

            if methodPrep != "None":
                # Adaptive threshold
                thrPrep = tools.CACHE_FindThresPrep(currentImage, 1000, float(thrPrepCoef)) 

                # Top-hat
                currentImage = morphology.white_tophat(currentImage, morphology.disk(4))
        
                # Pre-filtering
                filteredImage = median(currentImage, np.ones((3,3)))
                approxPoint = filteredImage > thrPrep
            else:
                approxPoint = np.ones_like(currentImage)
        
            if dispCurrentImage:
                imagePlaceholder.image(currentImage, use_column_width = True, caption = "Processed image")

            # Approximation 
            #BLOBs = tools.CACHE_ExponentialApproximationMask(
            #            currentImage, 1 / (np.arange(1.0, 7.1, 0.1) ** 2), approxPoint,
            #            False, int(fsize), float(thresCoefOld), 3)
            BLOBs = tools.randon_BLOBS(1000)

            st.session_state['BLOBs'] = BLOBs
            st.session_state['detected'] = True

            imageBLOBs = crsImage.convert("RGBA")
            draw = ImageDraw.Draw(imageBLOBs)
            for BLOB in BLOBs:                
                y, x, r = BLOB          
                draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

            imagePlaceholder.image(imageBLOBs, use_column_width = True, caption = "Detected nanoparticles")
            st.session_state['imageBLOBs'] = imageBLOBs
    
        # Info about detected nanoparticles
        if st.session_state['detected']:
            st.markdown(f"<p class = 'text'>Nanoparticles detected: <b>{st.session_state['BLOBs'].shape[0]}</b></p>", unsafe_allow_html=True)

        # Slider for comparing the results before and after detection
        if st.session_state['detected']:        
            st.checkbox("Comparison mode", key = 'comparison', help = help_str)

        # Saving
        if st.session_state['detected']:
            safeImgCol, safeBLOBCol = st.columns(2)

            with safeImgCol:
                temp = Image.new(mode="RGBA", size = st.session_state['imageBLOBs'].size)
                draw = ImageDraw.Draw(temp)
                for BLOB in st.session_state['BLOBs']:                
                    y, x, r = BLOB          
                    draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

                file = io.BytesIO()
                temp.save(file, format = "PNG")

                st.download_button(
                    label = "Download nanoparticles image",
                    data = file.getvalue(),
                    file_name = "processed-image.tif",
                    use_container_width  = True,
                    help = help_str
                )

            with safeBLOBCol:
                file = io.StringIO()
                csv.writer(file).writerows(st.session_state['BLOBs'])

                st.download_button(
                    label = "Download nanoparticles *.csv",
                    data = file.getvalue(),
                    file_name = "nanoparticles.csv",
                    use_container_width  = True,
                    help = help_str
                )
    # END right side


# TAB 2
with tabInfo:
    heightCol = 500

    if not st.session_state['detected']: 
         st.markdown(f"""<div class = 'about'>
                        Nanoparticle detection is necessary to calculate their mass.
                        Please go to "Detection" tab.
                    </div>""", unsafe_allow_html=True)
    else:
        # Dashboard structure
        db11, db12 = st.columns([4, 2])
        db21, db22, db23 = st.columns([4, 4, 4])

        additionalSTR = ''         
        radius_nm = st.session_state['BLOBs'][:, 2] * st.session_state['scale']
        
        # Particle size distribution
        with db11:
            with st.container(border = True, height = heightCol):                
                chart_col, set_col = st.columns([4, 2])
                
                minRadius = np.min(radius_nm)
                maxRadius = np.max(radius_nm)

                minVal, maxVal = st.session_state['chartRange']
                
                flagCustomRangeRadius = True
                if (minVal == 'min') and (maxVal != 'max'):
                    radiusFiltered = radius_nm[radius_nm <= float(maxVal)]
                elif (minVal != 'min') and (maxVal == 'max'):
                    radiusFiltered = radius_nm[radius_nm >= float(minVal)]
                elif (minVal != 'min') and (maxVal != 'max'):
                    radiusFiltered = radius_nm[(radius_nm >= float(minVal)) & (radius_nm <= float(maxVal))]
                else:
                    flagCustomRangeRadius = False
                    radiusFiltered = radius_nm                
                
                if flagCustomRangeRadius:
                    temp1 = minVal if (minVal != "min") else f'{minRadius:.1f}'
                    temp2 = maxVal if (maxVal != "max") else f'{maxRadius:.1f}'
                    additionalSTR = f" (from {temp1} to {temp2})"
    

                with set_col:
                    st.subheader("Settings chart")

                    uniqueRadius = ['min'] + [f'{x:.1f}' for x in np.unique(radius_nm)] + ['max']
                    st.select_slider("Select a range of particle radius",
                        options = uniqueRadius,
                        key = 'chartRange',
                        value = ('min', 'max'),
                        help = help_str
                    )

                    st.checkbox("Display distribution function?",
                        key = 'distView',
                        help = help_str
                    )

                    st.checkbox("Recalculation parameters?",
                        key = 'recalculation',
                        help = help_str
                    )
                # END set_col

                with chart_col:
                    
                    fig = ff.create_distplot(
                        [radiusFiltered], [''], bin_size = 0.25, curve_type = 'kde', histnorm = 'probability',
                        colors = ['green'], show_curve = st.session_state['distView'], show_rug = False
                    )

                    fig.update_layout(
                        margin = dict(l=10, r=25, t=45, b=5),
                        title = dict(text = "Distribution of particle radius" + additionalSTR, font = dict(size=27)),
                        xaxis_title_text = 'Radius, nm',
                        yaxis_title_text = 'Particle fraction',
                        showlegend = False
                    )
                    
                    fig.update_xaxes(range = [np.floor(minRadius), np.ceil(maxRadius)],
                        showgrid = True, gridwidth = 0.5, gridcolor = '#606060'
                    )
                    fig.update_traces(hoverinfo = "x", hovertemplate = "radius: %{x:.2} nm")

                    st.plotly_chart(fig, use_container_width = True)
                # END chart_col
        # END db11

        # Nanoparticle parameters
        with db12:
            with st.container(border = True, height = heightCol):                        
                # for Pd (palladium)
                materialDensity = 12.02 * 10**-15 # nanogram / nanometer   

                if st.session_state['scale'] is None:
                    st.markdown(f"""<div class = 'text'>The image scale could not be determined automatically!</div>""", unsafe_allow_html = True)
                else:                    
                    st.markdown(f"""<div class = 'text'>
                                    Estimated scale: <b>{st.session_state['scale']:.3} nm/px</b> 
                                </div>""", unsafe_allow_html = True)

                    if st.session_state['recalculation']:
                        radius_nm = radiusFiltered

                    st.subheader("Primary parameters")                    
                    st.markdown(f"""<div class = 'text'>
                                    Average radius: <b>{np.mean(radius_nm):0.3} nm</b> 
                                </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class = 'text'>
                                    Variance radius: <b>{np.var(radius_nm):0.3} nm</b> 
                                </div>""", unsafe_allow_html=True)


                    st.subheader("Secondary parameters")  
                    volumeParticls = 4 / 3 * np.pi * radius_nm ** 3
                    areaParticls =  np.sum(2 * np.pi * radius_nm ** 2)
                    massParticls = np.sum(volumeParticls * materialDensity)

                    st.markdown(f"""<div class = 'text'>
                                    Mass: <b>{massParticls:0.2e} nanograms</b> 
                                </div>""", unsafe_allow_html = True)                        
                    st.markdown(f"""<div class = 'text'>
                                    Volume: <b>{np.sum(volumeParticls):0.2e} nanometers<sup>3</sup></b> 
                                </div>""", unsafe_allow_html = True)                        
                    st.markdown(f"""<div class = 'text'>
                                    Area: <b>{areaParticls:0.2e} nanometers<sup>2</sup></b> 
                                </div>""", unsafe_allow_html = True)
                    

                    st.subheader("Secondary parameters (norm)", help = help_str)
                    imageArea = st.session_state['sizeImage'][0] * st.session_state['sizeImage'][1] * st.session_state['scale'] ** 2
                    
                    st.markdown(f"""<div class = 'text'>
                                    Mass: <b>{massParticls/imageArea:0.2e} nanograms/nanometers<sup>2</sup></b> 
                                </div>""", unsafe_allow_html = True)                        
                    st.markdown(f"""<div class = 'text'>
                                    Volume: <b>{np.sum(volumeParticls)/imageArea:0.2e} nanometers</b> 
                                </div>""", unsafe_allow_html = True)                        
                    st.markdown(f"""<div class = 'text'>
                                    Area: <b>{areaParticls/imageArea:0.2e}</b> 
                                </div>""", unsafe_allow_html = True)
        # END db12

        # ?
        with db21:
            with st.container(border = True, height = heightCol):
                uniformityMap = tools.uniformity(st.session_state['BLOBs'], st.session_state['sizeImage'], 10)
                fig = px.density_heatmap(uniformityMap)
                st.plotly_chart(fig, use_container_width = True)
        # END db21

        # ?
        with db22:
            with st.container(border = True, height = heightCol):
                pass
        # END db22

        # ?
        with db23:
            with st.container(border = True, height = heightCol):
                pass
        # END db23




# Articls
st.markdown("""<div class = 'cite'> <b>How to cite</b>:
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

# Footer
st.markdown("""<div class = 'footer'>
        Laboratory of Cognitive Technologies and Simulating Systems, Tula State University © 2025 (Email support: muwsik@mail.ru)
    </div>""", unsafe_allow_html = True)
