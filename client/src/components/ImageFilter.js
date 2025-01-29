import React, { useState, useEffect, useRef } from "react";

/** Hard-coded label set; adapt as needed. */
const ALL_LABELS = ["Person", "Car", "Tree", "Unknown"];

/** The base URL for your Flask server. Adjust if needed. */
const BASE_URL = "http://127.0.0.1:5000";

/**
 * PARENT COMPONENT:
 * Manages search, reference image, mode toggle, etc.
 */
function AnnotationApp() {
  // ============== SEARCH / STATE ==============
  const [mode, setMode] = useState("single"); // "single" or "multiple"
  const [searchQuery, setSearchQuery] = useState("");
  const [processOption, setProcessOption] = useState("");
  const [classFilter, setClassFilter] = useState("");

  const [images, setImages] = useState([]); // array of {id, filename, dataUrl, status}
  const [referenceImage, setReferenceImage] = useState(null);

  // ============== PAGINATION (for multiple mode) ==============
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // ============== EFFECTS & FETCHES ==============
  useEffect(() => {
    if (mode === "single") {
      // If you want immediate fetch on single mode:
      if (searchQuery || processOption || classFilter) {
        fetchImages();
        fetchReferenceImage(searchQuery);
      }
    } else {
      // multiple mode
      if (searchQuery || processOption || classFilter) {
        fetchImages();
        fetchReferenceImage(searchQuery);
      }
    }
    // eslint-disable-next-line
  }, [mode, page, pageSize]);

  /** Fetch images from your Flask backend. 
      We'll unify single vs multiple for demonstration. */
  async function fetchImages() {
    try {
      const classParam = classFilter
        ? `&class=${encodeURIComponent(classFilter)}`
        : "";
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
    }
  }

  /** Fetch reference image if any. */
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

  /** Discard handler (shared). */
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
        prev.map((img) =>
          img.id === imageId ? { ...img, status: "discarded" } : img
        )
      );
    } catch (err) {
      console.error("Error discarding image:", err);
    }
  }

  /** SEARCH SUBMIT => reset page, fetch images + reference. */
  async function handleSearchSubmit(e) {
    e.preventDefault();
    setPage(1);
    await fetchReferenceImage(searchQuery);
    fetchImages();
  }

  // ============== RENDER ==============
  return (
    <div style={{ margin: "20px" }}>
      <h1>Consolidated Annotation Viewer (Single + Multiple)</h1>

      {/* SEARCH FORM */}
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

      {/* MODE TOGGLE */}
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

      {/* RENDER REFERENCE IMAGE */}
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

      {/* CONDITIONALLY RENDER SINGLE OR MULTIPLE */}
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

/**
 * SINGLE MODE COMPONENT (up to 1200x1600),
 * draws one image at a time with rubber-band boxes.
 */
function ImageFilter({ images, onDiscard, allLabels, baseUrl }) {
  // We show images[currentIndex].
  const [currentIndex, setCurrentIndex] = useState(0);
  // Rubber-band states:
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [tempBox, setTempBox] = useState(null); // { x, y, w, h, label }
  const [boxes, setBoxes] = useState([]);       // final array

  const containerRef = useRef(null);
  const [naturalSize, setNaturalSize] = useState({ width: 400, height: 400 });

  // Label selection for single mode
  const [activeLabel, setActiveLabel] = useState(allLabels[0]);

  // If you want to fetch existing annotations for the single image,
  // do so when `currentIndex` changes. Then transform YOLO -> displayed.
  // Omitted for brevity. Similar to the "multiple" approach.

  // Move left / right
  function handlePrev() {
    setCurrentIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1));
    setBoxes([]);
    setTempBox(null);
  }
  function handleNext() {
    setCurrentIndex((prev) =>
      prev === images.length - 1 ? 0 : prev + 1
    );
    setBoxes([]);
    setTempBox(null);
  }

  function onImageLoad(e) {
    // NATURAL size
    const w = e.target.naturalWidth;
    const h = e.target.naturalHeight;
    setNaturalSize({ width: w, height: h });
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

      // Auto-save
      await autoSaveSingle(newBoxes);
    }
  }

  async function autoSaveSingle(finalBoxes) {
    if (!images || images.length === 0) return;
    const img = images[currentIndex];
    const { id, filename } = img;

    // Convert displayed -> YOLO
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dispW = rect.width;
    const dispH = rect.height;

    const rx = naturalSize.width / dispW;
    const ry = naturalSize.height / dispH;

    // Suppose label -> classId by index in allLabels
    const labelToClassId = {};
    allLabels.forEach((lbl, i) => {
      labelToClassId[lbl] = i;
    });

    const yoloBoxes = finalBoxes.map((b) => {
      const nx = b.x * rx;
      const ny = b.y * ry;
      const nw = b.w * rx;
      const nh = b.h * ry;

      // YOLO normalized
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
      boxes: yoloBoxes.map((b) => ({
        classId: b.classId,
        x_center: b.x_center,
        y_center: b.y_center,
        w: b.w,
        h: b.h,
      })),
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
    // optionally auto-save empty
    // autoSaveSingle([]);
  }

  if (images.length === 0) {
    return <p>No images found for single mode.</p>;
  }

  const currentImage = images[currentIndex];

  return (
    <div style={{ textAlign: "center" }}>
      {/* LABEL SELECTOR */}
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

      <div
        ref={containerRef}
        style={{
          position: "relative",
          maxWidth: "1200px",
          maxHeight: "1600px",
          margin: "0 auto 10px",
          border: "1px solid #ccc",
          cursor: "crosshair",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <img
          src={currentImage.dataUrl}
          alt={currentImage.filename}
          onLoad={onImageLoad}
          draggable="false"
          onDragStart={(e) => e.preventDefault()}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            pointerEvents: "none", // pass events to container
          }}
        />
        {/* Render existing boxes (all), highlight active label in a different color? */}
        {boxes.map((box, i) => {
          // color if matches active label
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

      <div style={{ marginBottom: "10px" }}>
        <button onClick={handlePrev} style={{ marginRight: "5px" }}>
          Prev
        </button>
        <button onClick={handleNext} style={{ marginRight: "5px" }}>
          Next
        </button>

        {/* Discard button => green if already discarded, else red */}
        {currentImage.status === "discarded" ? (
          <button style={{ backgroundColor: "green", color: "#fff" }}>
            Discarded
          </button>
        ) : (
          <button
            style={{ backgroundColor: "red", color: "#fff", marginRight: "5px" }}
            onClick={() => onDiscard(currentImage.id)}
          >
            Discard
          </button>
        )}

        <button onClick={handleClear}>Clear Boxes</button>
      </div>
    </div>
  );
}

/**
 * MULTIPLE MODE COMPONENT:
 * A grid of images, each 480x640. Fetch existing YOLO, show boxes, label dropdown in each image.
 */
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
  // For each image => array of final boxes in displayed coords
  const [boxesById, setBoxesById] = useState({});
  // temp box
  const [tempBoxById, setTempBoxById] = useState({});
  // is drawing?
  const [isDrawingById, setIsDrawingById] = useState({});
  const [startPointById, setStartPointById] = useState({});
  // natural size for YOLO conversion
  const [naturalSizeById, setNaturalSizeById] = useState({});
  // label selection per image
  const [selectedLabelById, setSelectedLabelById] = useState({});

  const containerRefs = useRef({});

  // We fix displayed size for each image
  const DISPLAY_W = 480;
  const DISPLAY_H = 640;

  // ============== FETCH EXISTING ANNOTATIONS ==============
  useEffect(() => {
    // whenever images changes, fetch annotations for each
    images.forEach((img) => {
      fetchAnnotationsForImage(img.id);
    });
  }, [images]);

  async function fetchAnnotationsForImage(imageId) {
    // GET /api/images/:id/annotations => YOLO
    // For example: { boxes: [ { classId, x_center, y_center, w, h }, ... ], labelMap: { "0": "Person", ... } }
    try {
      const res = await fetch(`${baseUrl}/api/images/${imageId}/annotations`);
      if (!res.ok) {
        // might be no annotation => skip
        return;
      }
      const data = await res.json();
      const yoloBoxes = data.boxes || [];
      // We'll store them in boxesById, but can't convert until we know NAT size
      // We'll store them in a structure so we can convert after image loads
      setBoxesById((prev) => ({
        ...prev,
        [imageId]: yoloBoxes.map((yb) => ({
          yoloClassId: yb.classId,
          yoloXCenter: yb.x_center,
          yoloYCenter: yb.y_center,
          yoloW: yb.w,
          yoloH: yb.h,
          // displayed coords => to be computed later
          x: 0,
          y: 0,
          w: 0,
          h: 0,
          // guess label from allLabels:
          label: allLabels[yb.classId] || "Unknown",
        })),
      }));
    } catch (err) {
      console.error("Error fetching annotations for image=", imageId, err);
    }
  }

  // When the actual <img> loads, we do YOLO->display
  function handleImageLoad(e, imageId) {
    const natW = e.target.naturalWidth;
    const natH = e.target.naturalHeight;
    setNaturalSizeById((prev) => ({
      ...prev,
      [imageId]: { width: natW, height: natH },
    }));

    // convert any YOLO to displayed
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
          const xx = x_center_disp - w_disp / 2;
          const yy = y_center_disp - h_disp / 2;
          return { ...box, x: xx, y: yy, w: w_disp, h: h_disp };
        }
        return box;
      });
      return { ...prev, [imageId]: newArr };
    });
  }

  // DRAWING
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
    const rect = containerRefs.current[imageId]?.getBoundingClientRect();
    if (!rect) return;
    const start = startPointById[imageId];
    if (!start) return;
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    const x = Math.min(start.x, endX);
    const y = Math.min(start.y, endY);
    const w = Math.abs(endX - start.x);
    const h = Math.abs(endY - start.y);

    const lbl = selectedLabelById[imageId] || allLabels[0];
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

  // AUTO SAVE => displayed -> YOLO
  async function autoSaveMultiple(imageId, finalBoxes) {
    const nat = naturalSizeById[imageId] || { width: 1, height: 1 };
    const ratioX = nat.width / DISPLAY_W;
    const ratioY = nat.height / DISPLAY_H;
    // label->classId
    const labelMap = {};
    allLabels.forEach((lbl, i) => (labelMap[lbl] = i));

    const yoloBoxes = finalBoxes.map((b) => {
      // displayed => natural
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

  // CLEAR
  function handleClear(imageId) {
    setBoxesById((prev) => ({ ...prev, [imageId]: [] }));
    setTempBoxById((prev) => ({ ...prev, [imageId]: null }));
    // optionally auto-save empty
    // autoSaveMultiple(imageId, []);
  }

  function handleLabelChange(e, imageId) {
    setSelectedLabelById((prev) => ({ ...prev, [imageId]: e.target.value }));
  }

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
          const selectedLabel = selectedLabelById[imageId] || allLabels[0];

          return (
            <div key={imageId} style={{ border: "1px solid #ccc" }}>
              {/* The drawing container (480x640) */}
              <div
                ref={(el) => (containerRefs.current[imageId] = el)}
                style={{
                  position: "relative",
                  width: "480px",
                  height: "640px",
                  border: "1px solid #666",
                  cursor: "crosshair",
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
                  // highlight if box.label == selectedLabel
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

              {/* Controls inside the image display */}
              <div style={{ padding: "5px", textAlign: "center" }}>
                {/* Label dropdown */}
                <div style={{ marginBottom: "5px" }}>
                  <label>Label:&nbsp;</label>
                  <select
                    value={selectedLabel}
                    onChange={(e) => handleLabelChange(e, imageId)}
                  >
                    {allLabels.map((lbl) => (
                      <option key={lbl} value={lbl}>
                        {lbl}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Discard / Clear */}
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

/** Export the PARENT as default. 
    (ImageFilter & ImageGrid are defined above in the same file.) */
export default AnnotationApp;
