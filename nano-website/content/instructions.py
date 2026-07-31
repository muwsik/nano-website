import streamlit as st
import numpy as np
import datetime

def Header():
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
   
    
def About():
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from an electron microscope (SEM or TEM).
            <br>It will help you to detect nanoparticles in the image and calculate their statictics.
        </div>
    """, unsafe_allow_html = True)

    st.markdown("""
        <div class = 'about' style = "padding-bottom: 25px;">
            Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
        </div>
    """, unsafe_allow_html = True)


def DetectResult(countNP, time):
    st.markdown(f"""
        <p class = 'text'>
            Nanoparticles detected: <b>{countNP}</b> ({time//60}m : {time%60:02}s)
        </p>
    """, unsafe_allow_html = True)


def FiltrationResult(countNP):
    st.markdown(f"""
        <p class = 'text'>
            Nanoparticles after filtration: <b>{countNP}</b>
        </p>
    """, unsafe_allow_html = True)


def LabelUploderFileCVAT():    
    st.markdown(f"""
        Import <a href='https://app.cvat.ai/'>CVAT</a> data to calculate statistics (format 'CVAT for images 1.1')
    """, unsafe_allow_html = True)


def AboutSectionParticleParams():
    st.markdown(f"""
        <p class = 'text center'>
            The main parameters of nanoparticles can be represented as primary values: 
            the average diameters, its deviations, or a histogram of the diameters distribution. 
            <br>Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
            which can be normalized to the area of the SEM image.
        </p>
    """, unsafe_allow_html = True)

    
def MaterialDensity(name, value):
    st.markdown(f"""
        <div class = 'text' style = "font-size: 16px;">
            <b>{name}</b> density: <b>{value:.2e} ng/nm<sup>3</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def EstimatedScale(scale):
    if scale.unit == 'px':
        st.markdown(f"""
            <div class = 'text'>
                Couldn't automatically estimate scale! 
            </div>
        """, unsafe_allow_html = True)
    else:
        st.markdown(f"""
            <div class = 'text'>
                Estimated scale: <b>{scale.multiplier:.3f} {scale.unit}/px</b> 
            </div>
        """, unsafe_allow_html = True)


def DefMaterial(typeMaterial):
    st.markdown(f"""
        <div class = 'text'>
            Material: <b>{typeMaterial}</b> 
        </div>
    """, unsafe_allow_html = True)

    
def UserMaterial(typeMaterial, density):
    st.markdown(f"""
        <div class = 'text'>
            Material: <b>{typeMaterial} ({density:.2e} ng/nm<sup>3</sup>)</b> 
        </div>
    """, unsafe_allow_html = True)


def Quantity(p_count):
    st.markdown(f"""
        <div class = 'text'>
            Quantity: <b>{p_count}</b>
        </div>
    """, unsafe_allow_html = True)


def PrimaryParameters(diameters, unit):
    st.subheader("Primary parameters", anchor = False)

    st.markdown(f"""
        <div class = 'text'>
            Average diameter: <b>{np.mean(diameters):.2f} {unit}</b> 
        </div>
    """, unsafe_allow_html = True)

    st.markdown(f"""
        <div class = 'text'>
            Standart deviation diameters: <b>{np.std(diameters):.2f} {unit}</b> 
        </div>
    """, unsafe_allow_html = True)


def SecondaryParameters(paramsNP, unit):    
    st.subheader("Secondary parameters", anchor = False) 

    if unit != 'px':
        st.markdown(f"""
            <div class = 'text'>
                Mass: <b>{paramsNP["mass"]:0.2e} ng</b> 
            </div>
        """, unsafe_allow_html = True)   
    else:
        st.markdown(f"""
            <div class = 'text'>
                Mass: cannot be calculated (scale is unknown)
            </div>
        """, unsafe_allow_html = True)

    st.markdown(f"""
        <div class = 'text'>
            Volume: <b>{paramsNP["volume"]:0.2e} {unit}<sup>3</sup></b> 
        </div>
    """, unsafe_allow_html = True) 

    st.markdown(f"""
        <div class = 'text'>
            Area: <b>{paramsNP["area"]:0.2e} {unit}<sup>2</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def NormSecondaryParameters(paramsNP, unit):
    st.subheader("Secondary parameters (norm)",
        help = f"Values relative to the surface area is {paramsNP["imageArea"]:.2e} {unit}²",
        anchor = False
    )                    
               
    if unit == 'px':        
        st.markdown(f"""
            <div class = 'text'>
                Cannot be calculated (scale is unknown)
            </div>
        """, unsafe_allow_html = True)    
    else:
        st.markdown(f"""
            <div class = 'text'>
                Norm area: <b>{paramsNP["normArea"]:0.2f}</b> %
            </div>
        """, unsafe_allow_html = True)

        st.markdown(f"""
            <div class = 'text'>
                Norm mass: <b>{paramsNP["normMass"]:0.2e} ng/{unit}<sup>2</sup></b> 
            </div>
        """, unsafe_allow_html = True)


def AboutSectionSpatialDistribution():
    st.markdown(f"""
        <p class = 'text center'>
            Visual representation of nanoparticle-based statistics in image.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def AboutSectioQuality():
    st.markdown(f"""
        <p class = 'text center'>
            Quality evaluation of the automatically detected nanoparticles 
            based on the Jacquard measure and the expert's manual marking.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def LegendChartQuality():
    st.markdown("""
        <div style="text-align: center;">
            Types particles in chart:<br>
            <span class="particle-label blue">Detect by algorithm</span>
            <span class="particle-label green">Correctly identified by algorithm (TP)</span>
            <span class="particle-label red">Not identified by algorithm (FN)</span>
            <span class="particle-label orange">Identified but not confirmed by expert (FP)</span>
        </div>
    """, unsafe_allow_html=True)


def Guide1():
    st.subheader("Nanoparticle Detection and Filtering", anchor = False)  
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class='text'>All the following steps are performed on the <strong>Automatic Detection</strong> tab.</p>
            <ul>
                <li>
                    <p class='text'>
                        Step 1. Upload the original SEM image using the <strong>Browse File</strong> button.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        Step 2. Detect nanoparticles by clicking the <strong>Nanoparticles Detection</strong> button,
                        which becomes available after the image is uploaded. The detection process takes some time,
                        typically from a few seconds up to one minute.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        Step 3. After successful detection, the detected nanoparticles are filtered using the default
                        parameters. The filtered nanoparticles are displayed on the image as circles.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        Step 4. You can manually adjust the detection and filtering parameters by unchecking the
                        <strong>Use Default Settings</strong> option. <strong>IMPORTANT:</strong> To apply the detection
                        parameters, click the <strong>Nanoparticles Detection</strong> button again. The filtering
                        parameters are applied automatically.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)

    media_col.markdown(f"""
        <div class = 'text' style = "text-align: center;">
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide2():
    st.subheader("Working with Detection Results", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class='text'>The following features are available on the <strong>Automatic Detection</strong> tab after nanoparticle detection has been completed.</p>
            <ul>
                <li>
                    <p class='text'>
                        Detection results can be downloaded in several formats:
                        (1) Detected nanoparticles on a transparent background.
                        (2) Detected nanoparticles overlaid on the original image.
                        (3) A file containing the center coordinates and radius of each detected nanoparticle.
                        To download the desired result, select the appropriate option from the
                        <strong>What Results Should Be Saved?</strong> drop-down list and click the button on the right.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        If the image contains a scale bar and its physical length is specified, the image scale is
                        determined automatically. The detected scale can be displayed using the
                        <strong>Display Scale</strong> switch.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        The comparison mode is currently under development.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text center'>
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide3():
    st.subheader("CVAT Integration", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <ul>
                <li>
                    <p class='text'>
                        Detection results can be downloaded in a format supported by
                        <a href="https://app.cvat.ai/">CVAT</a>.
                        To do this, after completing nanoparticle detection on the
                        <strong>Automatic Detection</strong> tab, select
                        <strong>CVAT Task</strong> from the
                        <strong>What Results Should Be Saved?</strong> drop-down list
                        and click the button on the right. The downloaded backup archive
                        can then be used to create a new CVAT task.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        Annotations created in CVAT can be imported into the application.
                        First, export the backup archive of the corresponding CVAT task.
                        Then, on the <strong>Statistics Dashboard</strong> tab, select
                        <strong>Import from CVAT</strong> from the
                        <strong>Which Nanoparticles to Use</strong> drop-down list and
                        upload the backup archive using the corresponding file upload field.
                        If all requirements are met, all statistics sections will be
                        displayed automatically below.
                    </p>
                </li>
                <li>
                    <p class='text'>
                        More detailed information about CVAT integration is available in the
                        <a href="https://disk.yandex.ru/i/2U5wgJ8IjskREQ">
                            extended user guide
                        </a>.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text center'>
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide4():
    st.subheader("Quality Evaluation", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class='text'>All the following steps are performed on the <strong>Statistics Dashboard</strong> tab.</p>
            <ul>
                <li>
                    <p class='text'>
                        The <strong>Quality Evaluation</strong> section provides a quantitative assessment of nanoparticle
                        detection quality. First, an automatic detection result is required. It must either be available
                        on the <strong>Automatic Detection</strong> tab or imported as a CVAT backup archive in the
                        <strong>Global Dashboard Settings</strong> section. Next, upload the expert annotation in the
                        CVAT backup archive format to the corresponding field in the
                        <strong>Quality Evaluation</strong> section. If all requirements are met, the detection quality
                        will be displayed below as a percentage. A detailed description of the evaluation procedure
                        can be found in publication [2].
                    </p>
                </li>
                <li>
                    <p class='text'>
                        You can also visualize the detection quality assessment. To do this, enable the
                        <strong>Display Nanoparticles</strong> switch. An interactive plot will appear below,
                        displaying four types of nanoparticles. <strong>Blue</strong> nanoparticles are automatically
                        detected particles that match <strong>green</strong> nanoparticles annotated by the expert (TP).
                        <strong>Red</strong> nanoparticles are annotated by the expert but were not detected automatically (FN).
                        <strong>Yellow</strong> nanoparticles are automatically detected particles that were not confirmed
                        by the expert (FP).
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text' style = "text-align: center;">
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def HowCite():
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
        </div>
    """, unsafe_allow_html = True)

    tempCol[1].image(r"./nano-website/content/qr-code.svg",
        caption = "Web Nanoparticles QR-code",
        width = 'stretch'
    )   


def Footer():
    st.markdown(f"""
        <div class = 'footer'>
            Laboratory of Cognitive Technologies and Simulating Systems,
            Tula State University © {datetime.datetime.now().year} (E-mail: nanoweb.assist@gmail.com)
        </div>
    """, unsafe_allow_html = True)