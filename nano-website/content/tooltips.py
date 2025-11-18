
class Detect:
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
