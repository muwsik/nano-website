export const containerStyle = {
    position: "relative" as const,
    width: "100%",
    height: "100%"
}

export const svgStyle = {
    display: "block",
    width: "100%",
    height: "100%"
}


export const particleStyle = {
    fill: "none",
    stroke: "#00AA00",
    strokeWidth: 1,
    pointerEvents: "all" as const
}


export const tooltipStyle = {
    position: "absolute" as const,
    textAlign: "left" as const,
    whiteSpace: "nowrap" as const,
    userSelect: "none" as const,
    background: "white",
    border: "1px solid #888",
    borderRadius: "4px",
    padding: "6px 8px",
    fontSize: "12px",
    lineHeight: "1.15",
    boxShadow: "0 2px 6px rgba(0,0,0,0.5)",
    pointerEvents: "none" as const,
    zIndex: 1000
}


export const titleStyle = {
    fontWeight: "bold" as const,
    marginBottom: "4px"
}


export const rowStyle = {
    margin: "2px 0"
}

export const viewportStyle = {
    position: "relative" as const,
    width: "100%",
    height: "90dvh",
    overflow: "hidden" as const,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    userSelect: "none" as const,
    overscrollBehavior: "contain" as const,
    touchAction: "none" as const
}