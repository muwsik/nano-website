import streamlit as st

def Header():
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
   
    
def About():
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
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
