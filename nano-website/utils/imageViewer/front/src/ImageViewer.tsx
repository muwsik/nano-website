import { useRef, useState, useEffect, useLayoutEffect } from "react"

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
    particleStyle,
    viewportStyle
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

    const viewportRef = useRef<HTMLDivElement>(null)

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

    const [viewBox, setViewBox] = useState({
        x: 0,
        y: 0,
        width: 1,
        height: 1
    })

    useEffect(() => {
        if (imageWidth <= 0 || imageHeight <= 0)
            return

        setViewBox({
            x: 0,
            y: 0,
            width: imageWidth,
            height: imageHeight
        })

    }, [imageWidth, imageHeight])

    const [isDragging, setIsDragging] = useState(false)

    const [lastMouse, setLastMouse] = useState({
        x: 0,
        y: 0
    })

    const handleMouseDown = (event: React.MouseEvent) => {
        setIsDragging(true)

        setLastMouse({
            x: event.clientX,
            y: event.clientY
        })
    }

    const handleMouseMove = (event: React.MouseEvent) => {
        if (!isDragging)
            return

        const dx = event.clientX - lastMouse.x
        const dy = event.clientY - lastMouse.y

        setViewBox(prev => ({
            ...prev,
            x: prev.x - dx * prev.width / viewportRef.current!.clientWidth,
            y: prev.y - dy * prev.height / viewportRef.current!.clientHeight
        }))

        setLastMouse({
            x: event.clientX,
            y: event.clientY
        })
    }

    const handleMouseUp = () => {
        setIsDragging(false)
    }



    function showTooltip(
        event: React.PointerEvent<SVGCircleElement>,
        particle: Particle
    ) {

        if (!svgRef.current)
            return

        const container =
            viewportRef.current!
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

    
    const handleWheel = (event: React.WheelEvent) => {
        event.preventDefault()

        const rect = viewportRef.current!.getBoundingClientRect()

        const mouseX = event.clientX - rect.left
        const mouseY = event.clientY - rect.top

        const factor =
            event.deltaY < 0 ? 1 / 1.1 : 1.1

        setViewBox(prev => {
            const imageX = prev.x + mouseX * prev.width / rect.width
            const imageY = prev.y + mouseY * prev.height / rect.height

            const newWidth = prev.width * factor
            const newHeight = prev.height * factor

            return {
                x: imageX - mouseX * newWidth / rect.width,
                y: imageY - mouseY * newHeight / rect.height,
                width: newWidth,
                height: newHeight
            }
        })
    }

    const handleDoubleClick = () => {
        setViewBox({
            x: 0,
            y: 0,
            width: imageWidth,
            height: imageHeight
        })
    }

    
    return (
        <div
            ref={viewportRef}
            style={viewportStyle}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onDoubleClick={handleDoubleClick}
        >
            <div style={containerStyle}>
                <svg
                    ref={svgRef}
                    viewBox={`
                        ${viewBox.x}
                        ${viewBox.y}
                        ${viewBox.width}
                        ${viewBox.height}
                    `}
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
            </div>

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