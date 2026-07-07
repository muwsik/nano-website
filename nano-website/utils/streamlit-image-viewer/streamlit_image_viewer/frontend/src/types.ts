import type { FrontendState } from "@streamlit/component-v2-lib";

export interface Particle {
    id: number;
    x: number;
    y: number;
    diameter: number;
    projectionArea: number;
    volume: number;
    c0: number;
    approxError: number;
}


export interface TooltipPosition {
    x: number;
    y: number;
}


export interface ViewerMetadata {
    unit: string;
}

export interface ViewerState extends FrontendState {

}

export interface ViewerData {
    image: string;
    imageWidth: number;
    imageHeight: number;
    particles: Particle[];
    metadata: ViewerMetadata;
}