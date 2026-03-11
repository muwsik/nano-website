
DefaultToggle       = "Use the default settings recommended by the developers. The result is not always optimal"

TypeMicroscopePills = "Detected automatically when the image is first loaded"

class Detection:
    Brightness      = "The minimal brightness of nanoparticles and its surroundings in the image"
    Diameter        = "Hipotetical diameter of nanoparticles in pixels"
    Irregularities  = "Image areas that are formed as a result of highlighting the material surface"
    

class Filtarion:
    Brightness      = "Brightness in the central pixel of the nanoparticle"
    Diameter        = "Range of possible nanoparticle diameters in nanometers"
    Reliability     = "Higher values indicate better nanoparticle visibility and clearer separation from image background"
    Irregularities  = "The minimum area of the bright zone identified as an artifact of the background"


class Visualization:
    Scale           = "Show the estimated scale in image"
    Irregularities  = "The areas with background irregularities are colored red"
    Download        = "Click here to download the detection result in the specified format"


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
    If file is *.CSV, then each line format 'y, x, r' is a nanoparticle.
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

    OutScale        = """
        The image scale could not be determined automatically!
        Using default scale: 1.0 nm/px
    """

    NoResults       = """
        Nanoparticle detection is necessary to calculate their statistics.
        Please go to "Automatic detection" tab.
    """

    SmallResults    = """
        Nanoparticles after detection and filtration are less than 10! 
        Please go to the "Automatic detection" tab and change the detection,
        filtering settings or upload another EM image!
    """

    NowUsingCVAT    = """
        This section is designed for evaluating automated nanoparticle detection algorithms. 
        Currently using data imported from CVAT - please verify data accuracy before proceeding.
    """

    ReportLimit     = """
        The ability to submit a report is limited.
        Please contact us at nanoweb.assist@gmail.com
    """


class Options:
    TypeMicroscope  = {
        1: 'SEM', 
        2: 'TEM'
    }

    NanopartSize    = {
        0: "Small (1-10 pixels)",
        1: "Medium (5-15 pixels)",
        2: "Large (10-30 pixels)" 
    }

    Saving          = {
        0: "Particles on clear background (*.tif)",
        1: "Particles on EM-image (*.tif)",
        2: "Particles characteristics (*.csv)",
        3: "CVAT task (*.zip)"
    }

    NanoStatistic   = {
        0: "Automatically detected",
        1: "Import from CVAT",
        2: "None"
    }

    MaterialDensity = {
        0: "Palladium (Pd)",    # 12.02 * 10**-12 ng / nm^3
        1: "Cuprum (Cu)",       #  8.96 * 10**-12 ng / nm^3
        2: "Alloy 30% Au + 70% Pd (AuPd)",  # 14.10 * 10**-12 ng / nm^3
        3: "Alloy 70% Cu + 30% Zn (CuZn)",  #  8.42 * 10**-12 ng / nm^3
        4: "User density"
    }

    TypeChart       = {
        0: "Heatmap of particle count",
        1: "Visualization particles",
    }

