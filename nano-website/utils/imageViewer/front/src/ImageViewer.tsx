import { useRef, useState, useLayoutEffect } from "react"

import type {
    Particle,
    TooltipPosition,
    ViewerMetadata
} from "./types"

import {
    containerStyle,
    svgStyle,
    tooltipStyle,
    titleStyle,
    rowStyle,
    particleStyle
} from "./styles"


interface Props {
    image: string
    imageWidth: number
    imageHeight: number
    particles: Particle[]
    metadata: ViewerMetadata
}


function calculatePopupPosition(
    x: number,
    y: number,
    popupWidth: number,
    popupHeight: number,
    containerWidth: number,
    containerHeight: number,
    offset = 10
): TooltipPosition {

    let left = x + offset
    let top = y + offset

    if (left + popupWidth > containerWidth)
        left = x - popupWidth - offset
    if (left < offset)
        left = offset

    if (top + popupHeight > containerHeight)
        top = y - popupHeight - offset
    if (top < offset)
        top = offset

    return {
        x: left,
        y: top
    }
}


export default function ImageViewer({
    image,
    imageWidth,
    imageHeight,
    particles,
    metadata
}: Props) {

    const containerRef = useRef<HTMLDivElement>(null)

    const svgRef = useRef<SVGSVGElement>(null)

    const tooltipRef = useRef<HTMLDivElement>(null)

    const [selectedParticle, setSelectedParticle] =
        useState<Particle | null>(null)

    const [pointerPosition, setPointerPosition] =
        useState<TooltipPosition>({
            x: 0,
            y: 0
        })

    const [tooltipPosition, setTooltipPosition] =
        useState<TooltipPosition>({
            x: 0,
            y: 0
        })

    function showTooltip(
        event: React.PointerEvent<SVGCircleElement>,
        particle: Particle
    ) {

        if (!svgRef.current)
            return

        const container =
            containerRef.current!
                .getBoundingClientRect()

        setSelectedParticle(particle)

        setPointerPosition({
            x: event.clientX - container.left,
            y: event.clientY - container.top
        })
    }


    function hideTooltip() {
        setSelectedParticle(null)
    }

    useLayoutEffect(() => {
        if (
            !selectedParticle ||
            !tooltipRef.current ||
            !svgRef.current
        )
            return

        const tooltip =
            tooltipRef.current.getBoundingClientRect()

        const container =
            svgRef.current.parentElement!
                .getBoundingClientRect()

        setTooltipPosition(
            calculatePopupPosition(
                pointerPosition.x,
                pointerPosition.y,
                tooltip.width,
                tooltip.height,
                container.width,
                container.height
            )
        )
    }, [
        selectedParticle,
        pointerPosition
    ])

    
    return (
        <div
            ref={containerRef}
            style={containerStyle}
        >
            <svg
                ref={svgRef}
                viewBox={`0 0 ${imageWidth} ${imageHeight}`}
                preserveAspectRatio="xMidYMid meet"
                style={svgStyle}
            >
                <image
                    href={image}
                    x={0}
                    y={0}
                    width={imageWidth}
                    height={imageHeight}
                />

                {particles.map((particle) => (
                    <circle
                        key={particle.id}
                        cx={particle.x}
                        cy={particle.y}
                        r={particle.diameter / 2}
                        {...particleStyle}
                        onPointerEnter={(event) =>
                            showTooltip(event, particle)
                        }
                        onPointerLeave={hideTooltip}
                    />
                ))}
            </svg>

            {selectedParticle && (
                <div
                    ref={tooltipRef}
                    style={{
                        ...tooltipStyle,

                        left: tooltipPosition.x,
                        top: tooltipPosition.y
                    }}
                >
                    <div style={titleStyle}>
                        Particle info
                    </div>

                    <div style={rowStyle}>
                        Diameter:
                        {" "}
                        {selectedParticle.diameter.toFixed(1)}
                        {" "}
                        {metadata.unit}
                    </div>

                    <div style={rowStyle}>
                        Area (projection):
                        {" "}
                        {selectedParticle.projectionArea.toFixed(1)}
                        {" "}
                        {metadata.unit}²
                    </div>

                    <div style={rowStyle}>
                        Volume:
                        {" "}
                        {selectedParticle.volume.toFixed(1)}
                        {" "}
                        {metadata.unit}³
                    </div>

                    <div style={rowStyle}>
                        Brightness:
                        {" "}
                        {selectedParticle.c0.toFixed(0)}
                    </div>

                    <div style={rowStyle}>
                        Reliability:
                        {" "}
                        {(1 - selectedParticle.approxError).toFixed(2)}
                    </div>
                </div>
            )}
        </div>
    )
}