export interface Particle {
    id: number
    x: number
    y: number
    diameter: number
    projectionArea: number
    volume: number
    c0: number
    approxError: number
}


export interface TooltipPosition {
    x: number
    y: number
}


export interface ViewerMetadata {
    unit: string
}


export interface ViewerData {
    image: string
    image_width: number
    image_height: number
    particles: Particle[]
    metadata: ViewerMetadata
}