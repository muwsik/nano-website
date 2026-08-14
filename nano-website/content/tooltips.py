NoneInfo            = "A hint will appear soon.  \n"

DefaultToggle       = "Use the default settings recommended by the developers.  \n**The result is not always optimal**"

TypeMicroscopePills = "Detected automatically when the image is first loaded.  \n"

class Detection:
    Brightness      = "The minimal brightness of nanoparticles and its surroundings in the image.  \n"
    Diameter        = "Hipotetical diameter of nanoparticles in pixels.  \n"
    

class Filtration:
    Brightness      = "Brightness in the central pixel of the nanoparticle.  \n"
    Diameter        = "Range of possible nanoparticle diameters.  \n"
    Reliability     = "Higher values indicate better nanoparticle visibility  \n and clearer separation from image background.  \n"


class Visualization:
    Scale           = "Show the estimated scale in image.  \n"
    Irregularities  = "The areas with background irregularities are colored red.  \n"
    Download        = "Click here to download the detection result in the specified format.  \n"


NanopartSelectbox   = """
    Which results will be used to calculate the statistics.  
    If the value is 'None', statistics are not calculated
"""


class Distribution:
    Function        = "Show the nanoparticle diameter distribution function"
    Normalize       = "The values of the vertical axis will be as a percentage of the total number of particles"
    Selection       = "Statistics will be calculated for the column selected on the graph"
    Step            = "Step for constructing a histogram of the nanoparticle diameter distribution"
    Download        = "Uploading chart data for self-charting"


ExpertFileUploader  = """ 
    If file is *.ZIP, it must match the form CVAT for image 1.1.
"""


class Warnings:
    DetectSettings  = """
        The detection settings have been changed. 
        To accept the new settings, click the button "Nanoparticles detection"!
    """

    NoFoundNanos    = """
        Nanoparticles not found!
        Please change the detection settings or upload another EM image!
    """

    FiltrSettings   = """
        There are no nanoparticles satisfying the filtration settings!
        Please change the filtering settings!
    """

    OutScale        = "The image scale could **not** be estimated automatically!"

    NoResults       = """
        Nanoparticle detection is necessary to calculate their statistics.
        Please go to "Automatic detection" tab.
    """

    NoResultsCVAT   = """
        There are no labeled data that could be interpreted as nanoparticles.
    """

    BadFileFormat   = """
        The data format in the file does not match 'CVAT for image 1.1'
    """

    SmallResults    = """
        Nanoparticles after detection and filtration are less than 10! 
        Please go to the "Automatic detection" tab and change the detection,
        filtering settings or upload another EM image!
    """

    ReportLimit     = """
        The ability to submit a report is limited.
        Please contact us at nanoweb.assist@gmail.com
    """


class Options:
    TypeMicroscope  = {
        1: 'SEM-image', 
        2: 'TEM-image'
    }

    NanopartSize    = {
        0: "Small (1-10 pixels)",
        1: "Medium (5-15 pixels)",
        2: "Large (10-30 pixels)" 
    }

    Saving          = {
        0: "Particles on clear background (*.tif)",
        1: "Particles on EM-image (*.tif)",
        2: "Particles parameters (*.csv)",
        3: "CVAT task (*.zip)"
    }

    NanoStatistic   = {
        0: "Automatically detected",
        1: "Import from CVAT",
        2: "None"
    }

    MaterialName = {
        0: "Palladium (Pd)",                # 12.02 * 10**-12 ng / nm^3
        1: "Cuprum (Cu)",                   #  8.96 * 10**-12 ng / nm^3
        2: "Alloy 30% Au + 70% Pd (AuPd)",  # 14.10 * 10**-12 ng / nm^3
        3: "Alloy 70% Cu + 30% Zn (CuZn)",  #  8.42 * 10**-12 ng / nm^3
        4: "Custom"
    }

    MaterialDensity = {
        0: 12.02 * 10**-12,  # ng / nm^3
        1: 8.96 * 10**-12,   # ng / nm^3
        2: 14.10 * 10**-12,  # ng / nm^3
        3: 8.42 * 10**-12,   # ng / nm^3
    }

    TypeChart       = {
        0: "Heatmap of particle count",
        1: "Visualization particles",
    }

