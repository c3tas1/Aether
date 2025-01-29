import React, { useState, useEffect, useRef } from "react";

// Hard-coded labels for demonstration:
const ALL_LABELS = ["Person", "Car", "Tree", "Unknown"];

// Adjust if needed for your Flask server base URL
const BASE_URL = "http://127.0.0.1:5000";

/**
 * PARENT COMPONENT
 * Manages search, reference image, mode toggle, etc.
 */
function AnnotationApp() {
  // -------------- State --------------
  const [mode, setMode] = useState("single"); // "single" | "multiple"
  const [searchQuery, setSearchQuery] = useState("");
  const [processOption, setProcessOption] = useState("");
  const [classFilter, setClassFilter] = useState("");

  const [images, setImages] = useState([]); // array of { id, filename, dataUrl, status }
  const [referenceImage, setReferenceImage] = useState(null);

  // Pagination for multiple mode
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // -------------- Effects --------------
  useEffect(() => {
    // If user typed a search, fetch images + reference
    if (mode === "single") {
      if (searchQuery || processOption || classFilter) {
        fetchReferenceImage(searchQuery);
        fetchImages();
      }
    } else {
      // multiple mode
      if (searchQuery || processOption || classFilter) {
        fetchReferenceImage(searchQuery);
        fetchImages();
      }
    }
    // eslint-disable-next-line
  }, [mode, page, pageSize]);

  // -------------- Fetch Logic --------------
  async function fetchImages() {
    try {
      const classParam = classFilter ? `&class=${encodeURIComponent(classFilter)}` : "";
      let endpoint = "";
      if (mode === "single") {
        endpoint = `${BASE_URL}/api/images/single?search=${encodeURIComponent(
          searchQuery
        )}&process=${processOption}${classParam}&page=${page}&per_page=${pageSize}`;
      } else {
        endpoint = `${BASE_URL}/api/images/multiple?search=${encodeURIComponent(
          searchQuery
        )}&process=${processOption}${classParam}&page=${page}&limit=${pageSize}`;
      }

      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error(`fetchImages failed with status ${res.status}`);
      }
      const data = await res.json();
      if (Array.isArray(data)) {
        const mapped = data.map((item) => ({
          id: item.id,
          filename: item.filename,
          dataUrl: `data:image/jpeg;base64,${item.base64}`,
          status: item.status || "",
        }));
        setImages(mapped);
      } else {
        setImages([]);
      }
    } catch (err) {
      console.error("Error fetching images:", err);
      setImages([]);
    }
  }

  async function fetchReferenceImage(query) {
    if (!query) {
      setReferenceImage(null);
      return;
    }
    try {
      const refRes = await fetch(
        `${BASE_URL}/api/reference_image?search=${encodeURIComponent(query)}`
      );
      if (!refRes.ok) {
        setReferenceImage(null);
        return;
      }
      const refData = await refRes.json();
      if (refData.base64) {
        const dataUrl = `data:image/jpeg;base64,${refData.base64}`;
        setReferenceImage(dataUrl);
      } else {
        setReferenceImage(null);
      }
    } catch (err) {
      console.error("Error fetching reference image:", err);
      setReferenceImage(null);
    }
  }

  // -------------- Discard --------------
  async function handleDiscard(imageId) {
    try {
      const res = await fetch(`${BASE_URL}/api/images/${imageId}/discard`, {
        method: "PUT",
      });
      if (!res.ok) {
        throw new Error(`Discard failed: ${res.status}`);
      }
      // update local
      setImages((prev) =>
        prev.map((img) => (img.id === imageId ? { ...img, status: "discarded" } : img))
      );
    } catch (err) {
      console.error("Error discarding image:", err);
    }
  }

  // -------------- Search Submit --------------
  async function handleSearchSubmit(e) {
    e.preventDefault();
    setPage(1);
    await fetchReferenceImage(searchQuery);
    fetchImages();
  }

  // -------------- Render --------------
  return (
    <div style={{ margin: "20px" }}>
      <h1>Annotation Viewer (Single + Multiple) w/ Existing Annotations</h1>

      {/* Search Form */}
      <form onSubmit={handleSearchSubmit} style={{ marginBottom: "20px" }}>
        <label>Search:</label>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ marginLeft: "5px", marginRight: "10px" }}
        />

        <label>Class Filter:</label>
        <input
          type="text"
          value={classFilter}
          onChange={(e) => setClassFilter(e.target.value)}
          style={{ marginLeft: "5px", marginRight: "10px" }}
        />

        <label>Process:</label>
        <select
          value={processOption}
          onChange={(e) => setProcessOption(e.target.value)}
          style={{ marginLeft: "5px" }}
        >
          <option value="">All</option>
          <option value="free">Free</option>
          <option value="paid">Paid</option>
        </select>

        <button type="submit" style={{ marginLeft: "10px" }}>
          Search
        </button>
      </form>

      {/* Mode Toggle */}
      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={() => setMode("single")}
          style={{ fontWeight: mode === "single" ? "bold" : "normal" }}
        >
          Single Mode
        </button>
        <button
          onClick={() => setMode("multiple")}
          style={{ fontWeight: mode === "multiple" ? "bold" : "normal" }}
        >
          Multiple Mode
        </button>
      </div>

      {/* Reference Image */}
      <div style={{ marginBottom: "20px", textAlign: "center" }}>
        {referenceImage ? (
          <div
            style={{
              display: "inline-block",
              border: "1px solid #ccc",
              width: "200px",
              height: "200px",
            }}
          >
            <img
              src={referenceImage}
              alt="Reference"
              style={{ maxWidth: "100%", maxHeight: "100%" }}
            />
          </div>
        ) : (
          <div
            style={{
              display: "inline-flex",
              border: "1px solid #ccc",
              width: "200px",
              height: "200px",
              justifyContent: "center",
              alignItems: "center",
              color: "red",
            }}
          >
            No reference image
          </div>
        )}
      </div>

      {/* Conditionally render Single or Multiple */}
      {mode === "single" ? (
        <ImageFilter
          images={images}
          onDiscard={handleDiscard}
          allLabels={ALL_LABELS}
          baseUrl={BASE_URL}
        />
      ) : (
        <ImageGrid
          images={images}
          onDiscard={handleDiscard}
          page={page}
          setPage={setPage}
          pageSize={pageSize}
          setPageSize={setPageSize}
          allLabels={ALL_LABELS}
          baseUrl={BASE_URL}
        />
      )}
    </div>
  );
}

/** SINGLE MODE COMPONENT */
function ImageFilter({ images, onDiscard, allLabels, baseUrl }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  // Rubber-band states
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [tempBox, setTempBox] = useState(null); // { x, y, w, h, label }

  // Final displayed boxes for the current image
  const [boxes, setBoxes] = useState([]);

  // NATURAL size for YOLO conversion
  const [naturalSize, setNaturalSize] = useState({ width: 400, height: 400 });

  // Label selection for single mode
  const [activeLabel, setActiveLabel] = useState(allLabels[0]);

  // We'll store YOLO boxes from the server (not yet in displayed coords)
  const [pendingYolo, setPendingYolo] = useState([]);

  const containerRef = useRef(null);

  // -------------- ALWAYS call this effect --------------
  useEffect(() => {
    // If no images, reset state and skip
    if (images.length === 0) {
      setBoxes([]);
      setTempBox(null);
      setPendingYolo([]);
      return;
    }

    // We have images => fetch for the current index
    const currentImage = images[currentIndex];
    fetchSingleAnnotations(currentImage.id);

    // Reset displayed boxes
    setBoxes([]);
    setTempBox(null);
  }, [images, currentIndex]);

  async function fetchSingleAnnotations(imageId) {
    try {
      const res = await fetch(`${baseUrl}/api/images/${imageId}/annotations`);
      if (!res.ok) {
        // If 404 or no annotation, just skip
        setPendingYolo([]);
        return;
      }
      const data = await res.json();
      const yoloBoxes = data.boxes || [];
      // Convert each YOLO => add label from classId
      const withLabels = yoloBoxes.map((b) => {
        const label = allLabels[b.classId] || "Unknown";
        return { ...b, label };
      });
      setPendingYolo(withLabels);
    } catch (err) {
      console.error("Error fetching single annotations:", err);
      setPendingYolo([]);
    }
  }

  // On image load => convert YOLO => displayed
  function handleImageLoad(e) {
    if (images.length === 0) return;

    const natW = e.target.naturalWidth;
    const natH = e.target.naturalHeight;
    setNaturalSize({ width: natW, height: natH });

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dispW = rect.width;
    const dispH = rect.height;

    const ratioX = dispW / natW;
    const ratioY = dispH / natH;

    const displayed = pendingYolo.map((yb) => {
      const x_center_disp = yb.x_center * dispW;
      const y_center_disp = yb.y_center * dispH;
      const w_disp = yb.w * dispW;
      const h_disp = yb.h * dispH;
      const x = x_center_disp - w_disp / 2;
      const y = y_center_disp - h_disp / 2;
      return { x, y, w: w_disp, h: h_disp, label: yb.label };
    });
    setBoxes(displayed);
  }

  if (images.length === 0) {
    return <p>No images found for single mode.</p>;
  }
  const currentImage = images[currentIndex];
  const discarded = currentImage.status === "discarded";

  // Navigation
  function handlePrev() {
    setCurrentIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1));
  }
  function handleNext() {
    setCurrentIndex((prev) =>
      prev === images.length - 1 ? 0 : prev + 1
    );
  }

  // Drawing
  function handleMouseDown(e) {
    e.preventDefault();
    if (!containerRef.current) return;
    setIsDrawing(true);

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStartPoint({ x, y });
    setTempBox(null);
  }
  function handleMouseMove(e) {
    if (!isDrawing || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    const x = Math.min(startPoint.x, endX);
    const y = Math.min(startPoint.y, endY);
    const w = Math.abs(endX - startPoint.x);
    const h = Math.abs(endY - startPoint.y);
    setTempBox({ x, y, w, h, label: activeLabel });
  }
  async function handleMouseUp(e) {
    e.preventDefault();
    setIsDrawing(false);
    if (tempBox && tempBox.w > 5 && tempBox.h > 5) {
      const newBoxes = [...boxes, tempBox];
      setBoxes(newBoxes);
      setTempBox(null);
      // Auto-save => displayed => YOLO => POST
      await autoSaveSingle(newBoxes);
    }
  }

  async function autoSaveSingle(finalBoxes) {
    const { id, filename } = currentImage;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const dispW = rect.width;
    const dispH = rect.height;

    const rx = naturalSize.width / dispW;
    const ry = naturalSize.height / dispH;

    // label->classId
    const labelToClassId = {};
    allLabels.forEach((lbl, i) => {
      labelToClassId[lbl] = i;
    });

    const yoloBoxes = finalBoxes.map((b) => {
      const nx = b.x * rx;
      const ny = b.y * ry;
      const nw = b.w * rx;
      const nh = b.h * ry;

      const x_center = (nx + nw / 2) / naturalSize.width;
      const y_center = (ny + nh / 2) / naturalSize.height;
      const w_norm = nw / naturalSize.width;
      const h_norm = nh / naturalSize.height;

      return {
        classId: labelToClassId[b.label] ?? 0,
        x_center,
        y_center,
        w: w_norm,
        h: h_norm,
      };
    });

    const payload = {
      filename,
      width: naturalSize.width,
      height: naturalSize.height,
      boxes: yoloBoxes,
    };

    try {
      const res = await fetch(`${baseUrl}/api/images/${id}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error("Single auto-save error " + res.status);
      }
      console.log("Saved single annotation for imageId=", id);
    } catch (err) {
      console.error("Auto-save single error:", err);
    }
  }

  function handleClear() {
    setBoxes([]);
    setTempBox(null);
    // optionally autoSaveSingle([])
  }

  return (
    <div style={{ textAlign: "center" }}>
      {/* Label Selector */}
      <div style={{ marginBottom: "10px" }}>
        <label>Label:&nbsp;</label>
        <select
          value={activeLabel}
          onChange={(e) => setActiveLabel(e.target.value)}
        >
          {allLabels.map((lbl) => (
            <option key={lbl} value={lbl}>
              {lbl}
            </option>
          ))}
        </select>
      </div>

      {/* Navigation + Discard */}
      <div style={{ marginBottom: "10px" }}>
        <button onClick={handlePrev} style={{ marginRight: "10px" }}>
          Prev
        </button>
        <button onClick={handleNext} style={{ marginRight: "10px" }}>
          Next
        </button>

        {discarded ? (
          <button style={{ backgroundColor: "green", color: "white" }}>
            Discarded
          </button>
        ) : (
          <button
            style={{ backgroundColor: "red", color: "white" }}
            onClick={() => onDiscard(currentImage.id)}
          >
            Discard
          </button>
        )}

        <button onClick={handleClear} style={{ marginLeft: "10px" }}>
          Clear Boxes
        </button>
      </div>

      {/* SCROLLABLE container */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          maxWidth: "1200px",
          maxHeight: "80vh",
          margin: "0 auto 20px",
          border: "1px solid #ccc",
          cursor: "crosshair",
          overflow: "auto",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <img
          src={currentImage.dataUrl}
          alt={currentImage.filename}
          onLoad={handleImageLoad}
          draggable="false"
          onDragStart={(e) => e.preventDefault()}
          style={{
            width: "100%",
            height: "auto",
            objectFit: "contain",
            pointerEvents: "none",
          }}
        />

        {/* Render existing boxes */}
        {boxes.map((box, i) => {
          const color = box.label === activeLabel ? "lime" : "orange";
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: box.x,
                top: box.y,
                width: box.w,
                height: box.h,
                border: `2px solid ${color}`,
                pointerEvents: "none",
              }}
            />
          );
        })}

        {/* Rubber-band temp box */}
        {tempBox && (
          <div
            style={{
              position: "absolute",
              left: tempBox.x,
              top: tempBox.y,
              width: tempBox.w,
              height: tempBox.h,
              border: "2px dashed red",
              pointerEvents: "none",
            }}
          />
        )}
      </div>
    </div>
  );
}

/** MULTIPLE MODE COMPONENT */
function ImageGrid({
  images,
  onDiscard,
  page,
  setPage,
  pageSize,
  setPageSize,
  allLabels,
  baseUrl,
}) {
  const [boxesById, setBoxesById] = useState({});
  const [tempBoxById, setTempBoxById] = useState({});
  const [isDrawingById, setIsDrawingById] = useState({});
  const [startPointById, setStartPointById] = useState({});
  const [naturalSizeById, setNaturalSizeById] = useState({});
  const [selectedLabelById, setSelectedLabelById] = useState({});

  const containerRefs = useRef({});

  const DISPLAY_W = 480;
  const DISPLAY_H = 640;

  // fetch existing YOLO annotations after we have images
  useEffect(() => {
    images.forEach((img) => {
      fetchAnnotationsForImage(img.id);
    });
  }, [images]);

  async function fetchAnnotationsForImage(imageId) {
    try {
      const res = await fetch(`${baseUrl}/api/images/${imageId}/annotations`);
      if (!res.ok) {
        // no annotation or error => skip
        return;
      }
      const data = await res.json();
      const yoloBoxes = data.boxes || [];
      setBoxesById((prev) => ({
        ...prev,
        [imageId]: yoloBoxes.map((yb) => {
          const label = allLabels[yb.classId] || "Unknown";
          return {
            yoloClassId: yb.classId,
            yoloXCenter: yb.x_center,
            yoloYCenter: yb.y_center,
            yoloW: yb.w,
            yoloH: yb.h,
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            label,
          };
        }),
      }));
    } catch (err) {
      console.error("Error fetching annotations for image=", imageId, err);
    }
  }

  // On image load => convert YOLO -> displayed
  function handleImageLoad(e, imageId) {
    const natW = e.target.naturalWidth;
    const natH = e.target.naturalHeight;
    setNaturalSizeById((prev) => ({ ...prev, [imageId]: { width: natW, height: natH } }));

    setBoxesById((prev) => {
      const arr = prev[imageId] || [];
      const ratioX = DISPLAY_W / natW;
      const ratioY = DISPLAY_H / natH;
      const newArr = arr.map((box) => {
        if (box.yoloXCenter !== undefined) {
          const x_center_disp = box.yoloXCenter * DISPLAY_W;
          const y_center_disp = box.yoloYCenter * DISPLAY_H;
          const w_disp = box.yoloW * DISPLAY_W;
          const h_disp = box.yoloH * DISPLAY_H;
          const x = x_center_disp - w_disp / 2;
          const y = y_center_disp - h_disp / 2;
          return { ...box, x, y, w: w_disp, h: h_disp };
        }
        return box;
      });
      return { ...prev, [imageId]: newArr };
    });
  }

  // Drawing
  function handleMouseDown(e, imageId) {
    e.preventDefault();
    setIsDrawingById((p) => ({ ...p, [imageId]: true }));

    const rect = containerRefs.current[imageId]?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    setStartPointById((p) => ({ ...p, [imageId]: { x: sx, y: sy } }));
    setTempBoxById((p) => ({ ...p, [imageId]: null }));
  }

  function handleMouseMove(e, imageId) {
    if (!isDrawingById[imageId]) return;
    const container = containerRefs.current[imageId];
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const start = startPointById[imageId];
    if (!start) return;

    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    const x = Math.min(start.x, endX);
    const y = Math.min(start.y, endY);
    const w = Math.abs(endX - start.x);
    const h = Math.abs(endY - start.y);

    const lbl = selectedLabelById[imageId] || ALL_LABELS[0];
    setTempBoxById((p) => ({ ...p, [imageId]: { x, y, w, h, label: lbl } }));
  }

  async function handleMouseUp(e, imageId) {
    e.preventDefault();
    if (!isDrawingById[imageId]) return;

    setIsDrawingById((p) => ({ ...p, [imageId]: false }));

    const temp = tempBoxById[imageId];
    if (!temp || temp.w < 5 || temp.h < 5) return;

    setBoxesById((prev) => {
      const oldArr = prev[imageId] || [];
      const newArr = [...oldArr, temp];
      autoSaveMultiple(imageId, newArr);
      return { ...prev, [imageId]: newArr };
    });
    setTempBoxById((p) => ({ ...p, [imageId]: null }));
  }

  // Save => displayed -> YOLO
  async function autoSaveMultiple(imageId, finalBoxes) {
    const nat = naturalSizeById[imageId] || { width: 1, height: 1 };
    const ratioX = nat.width / DISPLAY_W;
    const ratioY = nat.height / DISPLAY_H;

    const labelMap = {};
    ALL_LABELS.forEach((lbl, i) => (labelMap[lbl] = i));

    const yoloBoxes = finalBoxes.map((b) => {
      const nx = b.x * ratioX;
      const ny = b.y * ratioY;
      const nw = b.w * ratioX;
      const nh = b.h * ratioY;

      const x_center = (nx + nw / 2) / nat.width;
      const y_center = (ny + nh / 2) / nat.height;
      const w_norm = nw / nat.width;
      const h_norm = nh / nat.height;

      const cid = labelMap[b.label] ?? 0;
      return {
        classId: cid,
        x_center,
        y_center,
        w: w_norm,
        h: h_norm,
      };
    });

    const img = images.find((x) => x.id === imageId);
    if (!img) return;

    const payload = {
      filename: img.filename,
      width: nat.width,
      height: nat.height,
      boxes: yoloBoxes,
    };

    try {
      const res = await fetch(`${baseUrl}/api/images/${imageId}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error("autoSaveMultiple error " + res.status);
      }
      console.log("Saved multiple annotation for imageId=", imageId);
    } catch (err) {
      console.error("Error saving multiple:", err);
    }
  }

  function handleClear(imageId) {
    setBoxesById((prev) => ({ ...prev, [imageId]: [] }));
    setTempBoxById((prev) => ({ ...prev, [imageId]: null }));
    // optionally autoSaveMultiple(imageId, []);
  }

  function handleLabelChange(e, imageId) {
    setSelectedLabelById((prev) => ({ ...prev, [imageId]: e.target.value }));
  }

  // -------------- Render --------------
  return (
    <div>
      {/* PAGE CONTROL */}
      <div style={{ marginBottom: "10px", textAlign: "center" }}>
        <label>Page Size: </label>
        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(parseInt(e.target.value, 10));
            setPage(1);
          }}
        >
          <option value={25}>25</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>

        <button
          onClick={() => setPage((p) => (p > 1 ? p - 1 : 1))}
          style={{ marginLeft: "5px" }}
        >
          Prev
        </button>
        <span style={{ margin: "0 5px" }}>Page {page}</span>
        <button onClick={() => setPage((p) => p + 1)}>Next</button>
      </div>

      {images.length === 0 && <p style={{ textAlign: "center" }}>No images found.</p>}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "10px",
          justifyItems: "center",
        }}
      >
        {images.map((img) => {
          const imageId = img.id;
          const isDiscarded = img.status === "discarded";

          const finalBoxes = boxesById[imageId] || [];
          const tempBox = tempBoxById[imageId] || null;
          const selectedLabel = selectedLabelById[imageId] || ALL_LABELS[0];

          return (
            <div
              key={imageId}
              style={{
                border: "1px solid #ccc",
                padding: "5px",
              }}
            >
              <div
                ref={(el) => (containerRefs.current[imageId] = el)}
                style={{
                  position: "relative",
                  width: "480px",
                  height: "640px",
                  border: "1px solid #666",
                  cursor: "crosshair",
                  marginBottom: "5px",
                }}
                onMouseDown={(e) => handleMouseDown(e, imageId)}
                onMouseMove={(e) => handleMouseMove(e, imageId)}
                onMouseUp={(e) => handleMouseUp(e, imageId)}
              >
                <img
                  src={img.dataUrl}
                  alt={img.filename}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                  }}
                  draggable="false"
                  onDragStart={(ev) => ev.preventDefault()}
                  onLoad={(ev) => handleImageLoad(ev, imageId)}
                />

                {/* existing boxes */}
                {finalBoxes.map((box, idx) => {
                  const color = box.label === selectedLabel ? "lime" : "orange";
                  return (
                    <div
                      key={idx}
                      style={{
                        position: "absolute",
                        left: box.x,
                        top: box.y,
                        width: box.w,
                        height: box.h,
                        border: `2px solid ${color}`,
                        pointerEvents: "none",
                      }}
                    />
                  );
                })}

                {/* temp box */}
                {tempBox && (
                  <div
                    style={{
                      position: "absolute",
                      left: tempBox.x,
                      top: tempBox.y,
                      width: tempBox.w,
                      height: tempBox.h,
                      border: "2px dashed red",
                      pointerEvents: "none",
                    }}
                  />
                )}
              </div>

              {/* Label selection + Discard + Clear */}
              <div style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "5px" }}>
                  <label>Label:&nbsp;</label>
                  <select
                    value={selectedLabel}
                    onChange={(e) => handleLabelChange(e, imageId)}
                  >
                    {ALL_LABELS.map((lbl) => (
                      <option key={lbl} value={lbl}>
                        {lbl}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <button
                    style={{
                      marginRight: "5px",
                      backgroundColor: isDiscarded ? "green" : "red",
                      color: "white",
                    }}
                    onClick={() => onDiscard(imageId)}
                  >
                    {isDiscarded ? "Discarded" : "Discard"}
                  </button>
                  <button onClick={() => handleClear(imageId)}>Clear Boxes</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default AnnotationApp;
