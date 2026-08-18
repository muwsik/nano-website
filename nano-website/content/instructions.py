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


def AboutSectionParticleParams():
    st.markdown(f"""
        <p class = 'text center'>
            The main parameters of nanoparticles can be represented as primary values: 
            the average diameters, its deviations, or a histogram of the diameters distribution. 
            <br>Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
            which can be normalized to the area of the SEM image.
        </p>
    """, unsafe_allow_html = True)

    
def MaterialDensity(typeMaterial = None, density = None):
    if typeMaterial is not None:
        st.markdown(f"""
            <div class = 'text'>
                Material: <b>{typeMaterial}</b> 
            </div>
        """, unsafe_allow_html = True)

    if density is not None:
        st.markdown(f"""
            <div class = 'text'>
                Density: <b>{density:.2e} ng/nm<sup>3</sup></b> 
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


def Quantity(p_count):
    st.markdown(f"""
        <div class = 'text'>
            Quantity: <b>{p_count}</b> particles
        </div>
    """, unsafe_allow_html = True)


def AboutPrimaryParameters():
    st.subheader("Primary parameters", anchor = False)


def MeanDiameter(mu, unit):
    st.markdown(f"""
        <div class = 'text'>
            Average diameter: <b>{mu:.2f} {unit}</b> 
        </div>
    """, unsafe_allow_html = True)


def StdDiameter(std, unit):
    st.markdown(f"""
        <div class = 'text'>
            Standart deviation diameters: <b>{std:.2f} {unit}</b> 
        </div>
    """, unsafe_allow_html = True)


def AboutSecondaryParameters():    
    st.subheader("Secondary parameters", anchor = False) 

def Mass(mass = None):
    if mass is not None:
        st.markdown(f"""
            <div class = 'text'>
                Mass: <b>{mass:0.2e} ng</b> 
            </div>
        """, unsafe_allow_html = True)   
    else:
        st.markdown(f"""
            <div class = 'text'>
                Mass: cannot be calculated (scale is unknown)
            </div>
        """, unsafe_allow_html = True)


def Volume(volume, unit):
    st.markdown(f"""
        <div class = 'text'>
            Volume: <b>{volume:0.2e} {unit}<sup>3</sup></b> 
        </div>
    """, unsafe_allow_html = True) 


def Area(area, unit):
    st.markdown(f"""
        <div class = 'text'>
            Area: <b>{area:0.2e} {unit}<sup>2</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def AboutNormSecondaryParameters(imageArea, unit):
    st.subheader("Secondary parameters (norm)",
        help = f"Values relative to the surface area is {imageArea:.1e} {unit}²",
        anchor = False
    )                    


def NormArea(nArea):
    st.markdown(f"""
        <div class = 'text'>
            Norm area: <b>{nArea:0.2f}</b> %
        </div>
    """, unsafe_allow_html = True)


def NormMass(mass = None): 
    if mass is None:        
        st.markdown(f"""
            <div class = 'text'>
                Cannot be calculated (scale is unknown)
            </div>
        """, unsafe_allow_html = True)    
    else:     
        st.markdown(f"""
            <div class = 'text'>
                Norm mass: <b>{mass:0.2e} ng/nm<sup>2</sup></b> 
            </div>
        """, unsafe_allow_html = True)


def AboutSectionSpatialDistribution():
    st.markdown(f"""
        <p class = 'text center'>
            Visual representation of nanoparticle-based statistics in image.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def AboutSectionQuality():
    st.markdown(f"""
        <p class = 'text center'>
            Quality estimation of the automatically detected nanoparticles 
            based on the Jacquard measure and the expert's manual marking.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def Quality(FN, FP, TP):    
    st.markdown(f"""
        <p class = 'text center'>
            Accuracy: {TP / (TP + FN + FP) * 100:.2f}%
            (TP {TP}; FN {FN}; FP {FP})
        </p>
    """, unsafe_allow_html = True)


def LegendChartQuality():
    st.markdown("""
        <div style="text-align: center; font-size: 15px">
            <span class="particle-label blue">Detect by algorithm (D)</span>
            <span class="particle-label green">Correctly identified by algorithm (TP)</span>
            <span class="particle-label red">Not identified by algorithm (FN)</span>
            <span class="particle-label orange">Identified but not confirmed by expert (FP)</span>
        </div>
    """, unsafe_allow_html=True)


def Guide1():
    st.header("Nanoparticle Detection and Filtering", anchor = False)
    st.markdown(f"""
        <div class = 'text-help'>
            <p>
                All the following steps are performed on the <strong>Automatic Detection</strong> tab.
            </p>
            <ul>
                <li>
                    <p>
                        Step 1. Upload the original SEM/TEM image using the <strong>Browse File</strong> button.
                    </p>
                </li>
                <li>
                    <p>
                        Step 2. Detect nanoparticles by clicking the <strong>Nanoparticles Detection</strong> button,
                        which becomes available after the image is uploaded. The detection process takes some time,
                        typically from a few seconds up to one minute.
                    </p>
                </li>
                <li>
                    <p>
                        Step 3. After successful detection (if particles are found), the detected nanoparticles are
                        filtered using the default parameters. The filtered nanoparticles are displayed on the image
                        as circles. If no particles are detected, a warning will appear and filtering will not be available.
                    </p>
                </li>
                <li>
                    <p>
                        Step 4. You can manually adjust the detection and filtering parameters by unchecking the
                        <strong>Use Default Settings</strong> option. <strong>IMPORTANT:</strong> To apply changes to the
                        detection parameters, click the <strong>Nanoparticles Detection</strong> button again.
                        Filtering parameters are applied automatically when sliders are moved.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)


def Guide2():
    st.header("Working with Detection Results", anchor = False)
    st.markdown(f"""
        <div class = 'text-help'>
            <p>
                The following features are available on the <strong>Automatic Detection</strong> tab after nanoparticle detection has been completed.
                </p>
            <ul>
                <li>
                    <p>
                        Detection results can be downloaded in several formats:
                        <ol>
                            <li>Detected nanoparticles on a transparent background (PNG image).</li>
                            <li>Detected nanoparticles overlaid on the original image (PNG image).</li>
                            <li>A CSV file containing the center coordinates and <strong>diameter</strong> (in pixels) of each detected nanoparticle.</li>
                            <li>CVAT Task backup (zip archive) for import into CVAT.</li>
                        </ol>
                        To download the desired result, select the appropriate option from the
                        <strong>What Results Should Be Saved?</strong> drop-down list and click the download button.
                    </p>
                </li>
                <li>
                    <p>
                        If the image contains a scale bar and its physical length is specified, the image scale is
                        determined automatically. The detected scale can be displayed on the image using the
                        <strong>Estimated scale</strong> toggle in the <strong>Visualization and saving results</strong> section.
                    </p>
                </li>
                <li>
                    <p>
                        To assess detection quality, use the <strong>Quality estimation</strong> section (see Guide 4).
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)


def Guide3():
    st.header("CVAT Integration", anchor = False)
    st.markdown(f"""
        <div class = 'text-help'>
            <ul>
                <li>
                    <p>
                        Detection results can be downloaded in a format supported by
                        <a href="https://app.cvat.ai/">CVAT</a>.
                        To do this, after completing nanoparticle detection on the
                        <strong>Automatic Detection</strong> tab, select
                        <strong>CVAT Task</strong> from the
                        <strong>What Results Should Be Saved?</strong> drop-down list
                        and click the download button. The downloaded backup archive
                        can then be used to create a new CVAT task.
                    </p>
                </li>
                <li>
                    <p>
                        Annotations created in CVAT can be imported into the application.
                        First, export the backup archive of the corresponding CVAT task.
                        Then, on the <strong>Statistics Dashboard</strong> tab, select
                        <strong>Import from CVAT</strong> from the
                        <strong>Which Nanoparticles to Use</strong> drop-down list.
                        A file upload field will appear – upload the backup archive (zip).
                        If all requirements are met, all statistics sections will be
                        displayed automatically below.
                    </p>
                </li>
                <li>
                    <p>
                        More detailed information about CVAT integration is available in the
                        <a href="https://disk.yandex.ru/i/2U5wgJ8IjskREQ">
                            extended user guide
                        </a>.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)


def Guide4():
    st.header("Quality Evaluation", anchor = False)
    st.markdown(f"""        
        <div class = 'text-help'>
            <p>
                Quality evaluation is performed on the <strong>Automatic Detection</strong> tab, after successful nanoparticle detection.
            </p>
            <ul>
                <li>
                    <p>
                        Open the <strong>Quality estimation</strong> expander (located in the left settings column).
                        Upload the expert annotation file in CVAT backup format (zip archive) using the
                        <strong>Expert markup file</strong> uploader.
                    </p>
                </li>
                <li>
                    <p>
                        Adjust the <strong>Jacquard measure threshold</strong> slider if needed. The results will be
                        computed automatically.
                    </p>
                </li>
                <li>
                    <p>
                        The main image will display colored circles indicating:
                        <ul>
                            <li><strong>Blue</strong> – all automatically detected nanoparticles (before comparison).</li>
                            <li><strong>Green (TP)</strong> – nanoparticles correctly matched with expert annotations.</li>
                            <li><strong>Red (FN)</strong> – expert annotations missed by automatic detection.</li>
                            <li><strong>Yellow (FP)</strong> – automatically detected particles not confirmed by expert.</li>
                        </ul>
                        A legend and counts for each category are shown below the image.
                    </p>
                </li>
                <li>
                    <p>
                        A detailed description of the evaluation procedure can be found in publication [2].
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)


def Guide5():
    st.header("Statistics Dashboard", anchor = False)
    st.markdown(f"""
        <div class = 'text-help'>
            <p>
                The <strong>Statistics Dashboard</strong> tab provides detailed analysis of detected or imported nanoparticles.
            </p>
            <ul>
                <li>
                    <p>
                        First, choose the data source in the <strong>Global dashboard settings</strong>:
                        <ul>
                            <li><strong>Use current detection</strong> – statistics will be calculated for the nanoparticles detected on the Automatic Detection tab.</li>
                            <li><strong>Import from CVAT</strong> – upload a CVAT backup archive to use its annotations.</li>
                            <li><strong>None</strong> – no statistics are displayed.</li>
                        </ul>
                    </p>
                </li>
                <li>
                    <p>
                        In the <strong>Particle parameters</strong> section you can:
                        <ul>
                            <li>View the distribution of particle diameters (histogram) with adjustable step and normalization.</li>
                            <li>Select the particle material to compute mass-related parameters.</li>
                            <li>See basic statistics (count, mean diameter, standard deviation, area, volume, mass).</li>
                            <li>Choose between heatmap or painted image visualization.</li>
                        </ul>
                    </p>
                </li>
                <li>
                    <p>
                        In the <strong>Nanoparticle spatial distribution</strong> section you can explore:
                        <ul>
                            <li>Fraction of empty subareas for different block sizes.</li>
                            <li>Distance to nearest nanoparticle distribution.</li>
                            <li>Average number of nanoparticles per unit area.</li>
                            <li>Average coverage by neighbors and average number of neighbors.</li>
                        </ul>
                    </p>
                </li>
                <li>
                    <p>
                        The <strong>Aggregate statistics</strong> block allows you to collect summary statistics for multiple images into a table, which can be exported.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)


def HowCite():
    tempCol = st.columns([8, 2], vertical_alignment = 'center')

    tempCol[0].markdown("""
        <div class = 'cite'>
            <b>How to cite</b>:
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