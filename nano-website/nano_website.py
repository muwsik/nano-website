import streamlit as st

import io, csv
import numpy as np
from skimage import morphology
from skimage.filters import median
from streamlit_image_comparison import image_comparison
from PIL import Image, ImageDraw
    
import style, tools, autoscale

# Run
# streamlit run .\nano-website\nano_website.py --server.enableXsrfProtection false

help_str = "be added soon"


def load_default_settings():
    st.session_state['imageUpload'] = False
    st.session_state['uploadedImage'] = None

    st.session_state['settingDefault'] = True
    st.session_state['param1'] = 16
    st.session_state['param2'] = (1.0, 3.6)
    st.session_state['param3'] = 0.15
    st.session_state['param-pre-1'] = 10
    

    st.session_state['detected'] = False
    st.session_state['BLOBs'] = None
    st.session_state['imageBLOBs'] = None

    st.session_state['comparison'] = False
    
    st.session_state['scale'] = None
    st.session_state['mass'] = None

    st.session_state['chartRange'] = ('min','max')
    st.session_state['distView'] = False

if 'imageUpload' not in st.session_state:
    load_default_settings()


# Header
style.set_style()
st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)

st.markdown("""<div class = 'about'>
                    Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
                    <br>It will help you to detect palladium nanoparticles in the image and calculate their statictics.
               </div>""", unsafe_allow_html = True)

st.markdown("""<div class = 'about'>
                    Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
               </div>""", unsafe_allow_html = True)


# Main content area
left, rigth = st.columns([8, 3])


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
            st.session_state['scale'] = None
            st.session_state['mass'] = None

        if (not st.session_state['detected']):
            imagePlaceholder.image(crsImage, use_container_width = True, caption = "Uploaded image")
        elif (not st.session_state['comparison']):
            imagePlaceholder.image(st.session_state['imageBLOBs'], use_container_width = True, caption = "Detected nanoparticles")        
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


with rigth:
    
    tabDetect, tabMass, tabStruct = st.tabs(["Detection", "Mass", "Structuring"])
    
    with tabDetect:
        st.header("Settings")

        st.checkbox("Use default settings?",
            disabled = not st.session_state['imageUpload'],
            key = 'settingDefault',
            help = "You need to upload an SEM image")
    
        # Preprocessing settings        
        with st.container(border = True):
            st.slider(
                "parametar thr_br",
                key = 'param-pre-1',
                disabled = st.session_state['settingDefault'],
            )


        # Filtering settings 
        with st.container(border = True):
            st.slider(
                "parametar thr_c0",
                key = 'param1',
                disabled = st.session_state['settingDefault'],
            )

            st.slider(
                "parametar thr_r",
                key = 'param2',
                min_value = 1.0,
                step = 0.1,
                max_value = 4.5,
                disabled = st.session_state['settingDefault'],
            )

            st.slider(
                "parametar thr_error",
                key = 'param3',
                min_value = 0.0,
                step = 0.01,
                max_value = 1.0,
                disabled = st.session_state['settingDefault'],
            )
        

        pushProcc = st.button("Nanoparticles detection",
            use_container_width = True,
            disabled = not st.session_state['imageUpload'],
            help = "You need to upload an SEM image")
    
        # Detecting
        if pushProcc:
            currentImage = np.copy(grayImage) 
         
            lowerBound = autoscale.findBorder(grayImage)        
            if (lowerBound is not None):
                currentImage = currentImage[:lowerBound, :] 

            params = {
                "sz_med" : 4,   # для предварительной обработки
                "sz_th":  4,    # для предварительной обработки (не надо равное 5 - кружки получаются большие) 
                "thr_br": int(st.session_state['param-pre-1']),   # порог яркости для отбрасывания лок. максимумов (Prefiltering)
                "min_dist": 5,  # минимальное расстояние между локальными максимумами при поиске локальных максимумов (Prefiltering)
                "wsize": 9,     # размер окна аппроксимации
                "rs": np.arange(1.0, 7.5, 0.1), # возможные радиусы наночастиц в пикселях
                "best_mode": 3, # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                "msk": 5,       # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                "met": 'exp',   # аппроксимирующая функция "exp" или "pol" 
                "npar": 2       # число параметров аппроксимации
            }

            st.write(st.session_state['param-pre-1'], st.session_state['param1'], st.session_state['param2'],st.session_state['param3'])


            # вычисляется только один раз при первом запуске детектирования
            helpMatrs, xy2 = tools.CACHE_HelpMatricesNew(params["wsize"], params["rs"])

            # вычисляется только один раз для одного и тогоже изображения
            lm = tools.CACHE_PrefilteringPoints(
                currentImage,
                params["sz_med"],
                params["sz_th"],
                params["min_dist"],
                params["thr_br"]
            )

            BLOBs, BLOBs_params = tools.CACHE_ExponentialApproximationMask_v3(
                currentImage,
                lm,
                xy2,
                helpMatrs,
                params
            )

            params_filter = {
                "thr_c0": st.session_state['param1'],
                "min_thr_r": st.session_state['param2'][0],   
                "max_thr_r": st.session_state['param2'][1], 
                "thr_error": st.session_state['param3'], 
            }

            BLOBs_filtered = tools.my_FilterBlobs(BLOBs, BLOBs_params, params_filter)


            st.session_state['BLOBs'] = BLOBs_filtered
            st.session_state['detected'] = True

            imageBLOBs = crsImage.convert("RGBA")
            draw = ImageDraw.Draw(imageBLOBs)
            for BLOB in BLOBs:                
                y, x, r = BLOB          
                draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

            imagePlaceholder.image(imageBLOBs, use_container_width = True, caption = "Detected nanoparticles")
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
    # END tabDetect

    with tabMass:        
        # Nanoparticle mass
        if st.session_state['detected']:            
            st.header("Settings coming soon...")

            densityPd = 12.02 * 10**-15 # nanograms / nanometer        
        
            flag = True
            if  (st.session_state['scale'] is None) or (st.session_state['mass'] is None):
                lowerBound = autoscale.findBorder(grayImage)
        
                if (lowerBound is not None):      
                    text = autoscale.findText(grayImage[lowerBound:, :])

                    scaleVal = autoscale.scale(text)

                    # Длина шкалы в пикселях
                    scaleLengthVal = autoscale.scaleLength(grayImage, lowerBound)

                    if (scaleVal is not None) and (scaleLengthVal is not None):
                        st.session_state['scale'] = scaleVal / scaleLengthVal 
                        radiusNM = st.session_state['BLOBs'][:, 2] * st.session_state['scale'];
                        V = 4 / 3 * np.pi * radiusNM ** 3
                        st.session_state['mass'] = np.sum(V * densityPd)
                    else:
                        flag = False
        
            if not flag:
                st.markdown(f"""<div class = 'text'>The image scale could not be determined automatically!</div>""", unsafe_allow_html=True)
            else:        
                st.markdown(f"""<div class = 'text'>
                                Estimated scale: <b>{st.session_state['scale']:0.4} nm/px</b> 
                            </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class = 'text'>
                                Mass of detected Pd-nanoparticles:<br> <b>{st.session_state['mass']:0.2e} nanograms</b> 
                            </div>""", unsafe_allow_html=True)
        else:            
            st.markdown(f"""<div class = 'text'>
                    Nanoparticle detection is necessary to calculate their mass.
                    Please go to "Detection" tab.
                </div>""", unsafe_allow_html=True)
    # END tabMass
    
    with tabStruct:
        if st.session_state['detected']:
            st.markdown(f"""<div class = 'text'>It's coming soon!</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class = 'text'>
                    Nanoparticle detection is necessary to calculate the structuring.
                    Please go to "Detection" tab.
                </div>""", unsafe_allow_html=True)

    # END tabStruct       

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



st.markdown("""<div class = 'footer'>
        Laboratory of Cognitive Technologies and Simulating Systems (LCTSS), Tula State University (TulSU) © 2024
        <br>Do you need help? Please e-mail to muwsik@mail.ru
    </div>""", unsafe_allow_html = True)
