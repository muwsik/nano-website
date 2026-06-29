import { Streamlit } from "streamlit-component-lib"
import { useEffect, useState } from "react"

import ImageViewer from "./ImageViewer"

import type {
    Particle,
    ViewerData,
    ViewerMetadata
} from "./types"


export default function App() {

    const [image, setImage] = useState("")

    const [particles, setParticles] = useState<Particle[]>([])

    const [imageWidth, setImageWidth] = useState(0)

    const [imageHeight, setImageHeight] = useState(0)

    const [metadata, setMetadata] = useState<ViewerMetadata>({
        unit: "px"
    })


    useEffect(() => {
        Streamlit.setComponentReady()

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                Streamlit.setFrameHeight()
            })
        })

        const handleRender = (event: Event) => {
            const args =
                (event as CustomEvent).detail.args as ViewerData

            setImage(args.image)
            setParticles(args.particles ?? [])
            setImageWidth(args.image_width)
            setImageHeight(args.image_height)
            setMetadata(args.metadata)
        }

        Streamlit.events.addEventListener(
            Streamlit.RENDER_EVENT,
            handleRender
        )

        return () => {
            Streamlit.events.removeEventListener(
                Streamlit.RENDER_EVENT,
                handleRender
            )
        }
    }, [])

    return (
        <ImageViewer
            image = {image}
            imageWidth = {imageWidth}
            imageHeight = {imageHeight}
            particles = {particles}
            metadata = {metadata}
        />
    )
}