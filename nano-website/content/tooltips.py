DefaultToggle       = "Use the default settings recommended by the developers. The result is not always optimal"

class Detection:
    Brightness      = "The average brightness of nanoparticles and its surroundings in the image"
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


NanoparticlesSelectbox = """Which results will be used to calculate the statistics.
                            If the value is 'None', statistics are not calculated"""


class Distribution:
    Function        = "Show the nanoparticle diameter distribution function"
    Normalize       = "The values of the vertical axis will be as a percentage of the total number of particles"
    Selection       = "Statistics will be calculated for the column selected on the graph"
    Step            = "Step for constructing a histogram of the nanoparticle diameter distribution"
    Download        = "Uploading chart data for self-charting"


ExpertFileUploader  = """If file is *.CSV, then each line format 'y, x, r' is a nanoparticle.
                         If file is *.ZIP, it must match the form CVAT for image 1.1."""

