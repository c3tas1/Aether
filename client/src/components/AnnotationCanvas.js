import React, { useRef, useEffect, useState } from "react";

/**
 * AnnotationCanvas: 
 * - Renders an <img> plus a <canvas> overlay for drawing boxes/polygons.
 * - Props:
 *    imageId          - ID of the image (used for referencing annotation in the parent)
 *    imageUrl         - base64 or normal URL
 *    annotationType   - "box" or "polygon"
 *    width, height    - canvas (and image) dimensions
 *    boxes            - array of { x, y, w, h }
 *    polygons         - array of [ { x, y }, { x, y }, ... ] (each polygon)
 *    onUpdateAnnotations(newBoxes, newPolygons) => void
 */
function AnnotationCanvas({
  imageId,
  imageUrl,
  annotationType,
  width,
  height,
  boxes,
  polygons,
  onUpdateAnnotations,
}) {
  const canvasRef = useRef(null);

  // For bounding-box
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [startPoint, setStartPoint] = useState(null);

  // For polygon
  const [currentPolyPoints, setCurrentPolyPoints] = useState([]);

  // DRAW the existing boxes/polygons + any "in-progress" shape
  const drawAll = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing bounding boxes (green)
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    (boxes || []).forEach((b) => {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    });

    // Draw polygons (semi-transparent red)
    ctx.fillStyle = "rgba(255,0,0,0.3)";
    ctx.strokeStyle = "red";
    (polygons || []).forEach((poly) => {
      if (poly.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(poly[0].x, poly[0].y);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i].x, poly[i].y);
      }
      // If you want polygons closed:
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    // If we have a partial polygon in progress
    if (annotationType === "polygon" && currentPolyPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(currentPolyPoints[0].x, currentPolyPoints[0].y);
      for (let i = 1; i < currentPolyPoints.length; i++) {
        ctx.lineTo(currentPolyPoints[i].x, currentPolyPoints[i].y);
      }
      ctx.stroke();
    }
  };

  // Re-draw whenever boxes/polygons/currentPoly changes
  useEffect(() => {
    drawAll();
    // eslint-disable-next-line
  }, [boxes, polygons, currentPolyPoints]);

  // BOUNDING BOX EVENTS
  const handleMouseDownBox = (e) => {
    setIsDrawingBox(true);
    const rect = e.currentTarget.getBoundingClientRect();
    setStartPoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleMouseMoveBox = (e) => {
    if (!isDrawingBox || !startPoint) return;
    // Re-draw existing, then draw a "live" rectangle
    drawAll();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    const rect = e.currentTarget.getBoundingClientRect();
    const currX = e.clientX - rect.left;
    const currY = e.clientY - rect.top;
    const w = currX - startPoint.x;
    const h = currY - startPoint.y;
    ctx.strokeRect(startPoint.x, startPoint.y, w, h);
  };

  const handleMouseUpBox = (e) => {
    if (!isDrawingBox || !startPoint) return;
    setIsDrawingBox(false);
    const rect = e.currentTarget.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    const w = endX - startPoint.x;
    const h = endY - startPoint.y;
    const newBox = { x: startPoint.x, y: startPoint.y, w, h };
    // Update parent's annotation
    onUpdateAnnotations([...boxes, newBox], polygons);
    setStartPoint(null);
  };

  // POLYGON EVENTS
  const handleClickPolygon = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Double-click => finish the polygon
    if (e.detail === 2) {
      // Only finalize if we have >= 3 points
      if (currentPolyPoints.length >= 2) {
        // Add the polygon
        onUpdateAnnotations(boxes, [...polygons, currentPolyPoints]);
      }
      setCurrentPolyPoints([]);
    } else {
      // Single click => add a point
      setCurrentPolyPoints((prev) => [...prev, { x: clickX, y: clickY }]);
    }
  };

  // Decide event handlers based on annotationType
  const handleMouseDown = (e) => {
    if (annotationType === "box") {
      handleMouseDownBox(e);
    }
  };
  const handleMouseMove = (e) => {
    if (annotationType === "box") {
      handleMouseMoveBox(e);
    }
  };
  const handleMouseUp = (e) => {
    if (annotationType === "box") {
      handleMouseUpBox(e);
    }
  };
  const handleClick = (e) => {
    if (annotationType === "polygon") {
      handleClickPolygon(e);
    }
  };

  return (
    <div style={{ position: "relative" }}>
      <img
        src={imageUrl}
        alt="Annotation"
        style={{
          width: `${width}px`,
          height: `${height}px`,
          objectFit: "contain",
          display: "block",
        }}
      />
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          cursor: annotationType === "box" ? "crosshair" : "default",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
      />
    </div>
  );
}

export default AnnotationCanvas;
