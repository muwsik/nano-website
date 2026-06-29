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

//// Function
function calculateTooltipPosition(
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


//// ImageViewer
export default function ImageViewer({
    image,
    imageWidth,
    imageHeight,
    particles,
    metadata
}: Props) {

    // Refs
    const viewportRef = useRef<HTMLDivElement>(null)
    const svgRef = useRef<SVGSVGElement>(null)
    const tooltipRef = useRef<HTMLDivElement>(null)


    // Tooltip
    const [selectedParticle, setSelectedParticle] = useState<Particle | null>(null)
    const [tooltipPosition, setTooltipPosition] = useState<TooltipPosition>({ x: 0, y: 0 })
    const [pointerPosition, setPointerPosition] = useState<TooltipPosition>({ x: 0, y: 0 })

    function handleParticleEnter(
        event: React.PointerEvent<SVGCircleElement>,
        particle: Particle
    ) {
        if (!svgRef.current)
            return

        const container = viewportRef.current!.getBoundingClientRect()
        setSelectedParticle(particle)
        setPointerPosition({
            x: event.clientX - container.left,
            y: event.clientY - container.top
        })
    }

    function hideTooltip() {
        setSelectedParticle(null)
    }


    // View
    const [viewBox, setViewBox] = useState({ x: 0, y: 0, width: 100, height: 100 })

    function fitToWindow() {
        setViewBox({
            x: 0,
            y: 0,
            width: imageWidth,
            height: imageHeight
        })
    }

    function panBy(dx: number, dy: number) {
        const viewport = viewportRef.current
        if (!viewport)
            return

        setViewBox(prev => ({
            ...prev,
            x: prev.x - dx * prev.width / viewport.clientWidth,
            y: prev.y - dy * prev.height / viewport.clientHeight
        }))
    }

    function zoomAt(mouseX: number, mouseY: number, factor: number, rWidth: number, rHeight: number) {
        setViewBox(prev => {
            const imageX = prev.x + mouseX * prev.width / rWidth
            const imageY = prev.y + mouseY * prev.height / rHeight

            const width = prev.width * factor
            const height = prev.height * factor

            return {
                x: imageX - mouseX * width / rWidth,
                y: imageY - mouseY * height / rHeight,
                width,
                height
            }
        })
    }


    // Mouse interaction
    const [isDragging, setIsDragging] = useState(false)
    const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 })

    const handleWheel = (event: React.WheelEvent) => {
        event.preventDefault()
        const rect = viewportRef.current!.getBoundingClientRect()

        const mouseX = event.clientX - rect.left
        const mouseY = event.clientY - rect.top
        const scaleFactor = event.deltaY < 0 ? 1 / 1.1 : 1.1

        zoomAt(mouseX, mouseY, scaleFactor, rect.width, rect.height)
    }

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

        panBy(event.clientX - lastMouse.x, event.clientY - lastMouse.y)

        setLastMouse({
            x: event.clientX,
            y: event.clientY
        })
    }

    const handleMouseUp = () => {
        setIsDragging(false)
    }


    // Initialize view when a new image is loaded
    useEffect(() => {
        if (imageWidth <= 0 || imageHeight <= 0)
            return

        fitToWindow()
    }, [imageWidth, imageHeight])

    // Update tooltip position
    useLayoutEffect(() => {
        if (
            !selectedParticle ||
            !tooltipRef.current ||
            !svgRef.current
        )
            return

        const tooltip = tooltipRef.current.getBoundingClientRect()
        const container = svgRef.current.parentElement!.getBoundingClientRect()

        setTooltipPosition(
            calculateTooltipPosition(
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
            ref={viewportRef}
            style={viewportStyle}

            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onDoubleClick={fitToWindow}
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
                                handleParticleEnter(event, particle)
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
                        {" "} {selectedParticle.diameter.toFixed(1)}
                        {" "} {metadata.unit}
                    </div>

                    <div style={rowStyle}>
                        Area (projection):
                        {" "} {selectedParticle.projectionArea.toFixed(1)}
                        {" "} {metadata.unit}²
                    </div>

                    <div style={rowStyle}>
                        Volume:
                        {" "} {selectedParticle.volume.toFixed(1)}
                        {" "} {metadata.unit}³
                    </div>

                    <div style={rowStyle}>
                        Brightness:
                        {" "} {selectedParticle.c0.toFixed(0)}
                    </div>

                    <div style={rowStyle}>
                        Reliability:
                        {" "} {(1 - selectedParticle.approxError).toFixed(2)}
                    </div>
                </div>
            )}
        </div>
    )
}