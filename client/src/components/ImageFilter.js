import React, { useState, useEffect, useRef } from "react";

// Define your backend base URL once here:
const BASE_URL = "http://127.0.0.1:5000";

// 16 class names
const CLASS_NAMES = [
  "person", "car", "dog", "cat", "bottle", "chair", "tv", "phone",
  "book", "cup", "laptop", "boat", "bird", "plane", "cow", "sheep"
];

function ImageFilter() {
  // Upload
  const [selectedFiles, setSelectedFiles] = useState([]);

  // Search & Mode
  const [searchQuery, setSearchQuery] = useState("");
  const [feeOption, setFeeOption] = useState(""); // can be "", "Obj Det", etc.
  const [mode, setMode] = useState("single"); // "single" or "multiple"

  // Images
  // Each image => { id, filename, dataUrl, status, boxes: [ {classId, x, y, w, h} ] }
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Pagination (multiple mode)
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Annotation states
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPt, setStartPt] = useState(null);

  // The user's chosen class ID (0..15)
  const [currentClassId, setCurrentClassId] = useState(0);

  // Canvas ref (single mode)
  const singleCanvasRef = useRef(null);

  // =================== 1) FILE UPLOAD ===================
  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) {
      alert("No files selected");
      return;
    }
    try {
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append("images", file);
      }
      const uploadUrl = `${BASE_URL}/api/upload`;  // Use BASE_URL
      const res = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Upload error " + res.status);
      }
      const result = await res.json();
      console.log("Upload result:", result);
      alert("Upload successful");
      setSelectedFiles([]);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload failed. Check console.");
    }
  };

  // =================== 2) FETCH IMAGES ===================
  const fetchImages = async () => {
    try {
      let endpoint = "";
      if (mode === "single") {
        endpoint = `${BASE_URL}/api/images/single?search=${searchQuery}&fee=${feeOption}&page=${page}&per_page=${pageSize}`;
      } else {
        endpoint = `${BASE_URL}/api/images/multiple?search=${searchQuery}&fee=${feeOption}&page=${page}&limit=${pageSize}`;
      }
      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error("Fetch error " + res.status);
      }
      const data = await res.json();
      // data => array of { id, filename, base64, status, boxes? }
      const mapped = (Array.isArray(data) ? data : []).map(item => ({
        id: item.id,
        filename: item.filename,
        dataUrl: "data:image/jpeg;base64," + item.base64,
        status: item.status || "",
        boxes: item.boxes || []
      }));
      setImages(mapped);
      setCurrentIndex(0);
    } catch (err) {
      console.error("Fetch images error:", err);
      alert("Error loading images. Check console.");
      setImages([]);
    }
  };

  // Re-fetch if certain states change
  useEffect(() => {
    if (searchQuery || feeOption) {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode, page, pageSize, feeOption]);

  // =================== 3) SINGLE MODE ANNOTATION ===================
  useEffect(() => {
    if (mode === "single") {
      drawSingleBoxes();
    }
    // eslint-disable-next-line
  }, [images, mode, currentIndex]);

  const drawSingleBoxes = () => {
    if (mode !== "single") return;
    const canvas = singleCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (images.length === 0) return;
    const img = images[currentIndex];
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    img.boxes.forEach(box => {
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      // label
      const label = CLASS_NAMES[box.classId] || `id=${box.classId}`;
      ctx.fillStyle = "lime";
      ctx.font = "14px Arial";
      ctx.fillText(label, box.x + 2, box.y + 14);
    });
  };

  const handleSingleMouseDown = (e) => {
    setIsDrawing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    setStartPt({ x: sx, y: sy });
  };
  const handleSingleMouseMove = (e) => {
    if (!isDrawing) return;
    drawSingleBoxes();
    const canvas = singleCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    ctx.strokeRect(startPt.x, startPt.y, mx - startPt.x, my - startPt.y);
  };
  const handleSingleMouseUp = (e) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = mx - startPt.x;
    const h = my - startPt.y;
    const newBox = {
      classId: currentClassId,
      x: startPt.x,
      y: startPt.y,
      w, h
    };
    const copy = [...images];
    copy[currentIndex].boxes.push(newBox);
    setImages(copy);
    setStartPt(null);
  };

  const saveSingleAnnotations = async () => {
    if (images.length === 0) return;
    const img = images[currentIndex];
    try {
      const body = {
        boxes: img.boxes,
        imageWidth: 400, // single mode
        imageHeight: 400
      };
      const annUrl = `${BASE_URL}/api/annotations/${img.filename}`;
      const res = await fetch(annUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error("Save error " + res.status);
      alert("Saved annotations for " + img.filename);
    } catch (err) {
      console.error("saveSingleAnnotations error:", err);
      alert("Save failed. See console.");
    }
  };

  const handlePrevImage = () => {
    setCurrentIndex(i => (i === 0 ? images.length - 1 : i - 1));
  };
  const handleNextImage = () => {
    setCurrentIndex(i => (i === images.length - 1 ? 0 : i + 1));
  };

  // =================== 4) MULTIPLE MODE THUMBNAILS ===================
  // We'll use a subcomponent for each thumbnail

  // =================== 5) Discard ===================
  const handleDiscard = async (img) => {
    try {
      const discardUrl = `${BASE_URL}/api/images/${img.id}/discard`;
      const res = await fetch(discardUrl, {
        method: "PUT"
      });
      if (!res.ok) throw new Error("Discard error " + res.status);
      const result = await res.json();
      alert("Discarded image_id: " + result.image_id);
      setImages(prev => prev.map(x => (x.id === img.id ? { ...x, status: "discarded" } : x)));
    } catch (err) {
      console.error("Discard error:", err);
      alert("Discard failed. See console.");
    }
  };

  // =================== 6) Searching Form ===================
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchImages();
  };

  return (
    <div style={{ margin: "20px" }}>
      <h1>React YOLOv7 Annotation (with Base URL variable)</h1>

      {/* ========== Upload Form ========== */}
      <h2>Upload Images or ZIP</h2>
      <form onSubmit={handleUploadSubmit} style={{ marginBottom: "20px" }}>
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          accept=".zip,image/*"
          style={{ marginRight: "10px" }}
        />
        <button type="submit">Upload</button>
      </form>

      {/* ========== Search Form ========== */}
      <form onSubmit={handleSearchSubmit} style={{ marginBottom: "10px" }}>
        <label>Search:</label>
        <input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ marginLeft: "5px", marginRight: "10px" }}
        />
        <label>Fee:</label>
        <select
          value={feeOption}
          onChange={(e) => setFeeOption(e.target.value)}
          style={{ marginLeft: "5px" }}
        >
          <option value="">All</option>
          <option value="Obj Det">Obj Det</option>
        </select>
        <button type="submit" style={{ marginLeft: "10px" }}>
          Search
        </button>
      </form>

      {/* ========== Mode Toggle ========== */}
      <div style={{ marginBottom: "10px" }}>
        <button
          onClick={() => setMode("single")}
          style={{ marginRight: "10px", fontWeight: mode === "single" ? "bold" : "normal" }}
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

      {/* ========== Class Selector ========== */}
      <div style={{ marginBottom: "10px" }}>
        <label>Class: </label>
        <select
          value={currentClassId}
          onChange={(e) => setCurrentClassId(Number(e.target.value))}
        >
          {CLASS_NAMES.map((name, idx) => (
            <option key={idx} value={idx}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {/* ========== SINGLE MODE ========== */}
      {mode === "single" && images.length > 0 && (
        <div
          style={{
            // Center the single-mode container
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              position: "relative",
              width: "25%", // approximate 1/4 scale
              margin: "auto",
            }}
          >
            {/* The image scaled to 25% width */}
            <img
              src={images[currentIndex].dataUrl}
              alt=""
              style={{
                display: "block",
                width: "100%",
                height: "auto",
              }}
            />
            {/* Canvas with 400x400 coordinate space, but visually scaled to 25% */}
            <canvas
              ref={singleCanvasRef}
              width={400}
              height={400}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%", // match the scaled image
                height: "auto",
                cursor: "crosshair",
              }}
              onMouseDown={handleSingleMouseDown}
              onMouseMove={handleSingleMouseMove}
              onMouseUp={handleSingleMouseUp}
            />
          </div>
          <div style={{ marginTop: "10px" }}>
            <button onClick={saveSingleAnnotations}>Save Annotations</button>{" "}
            <button onClick={() => handleDiscard(images[currentIndex])}>
              {images[currentIndex].status === "discarded" ? "Discarded" : "Discard"}
            </button>{" "}
            <button onClick={handlePrevImage}>Prev</button>{" "}
            <button onClick={handleNextImage}>Next</button>
          </div>
        </div>
      )}
      {mode === "single" && images.length === 0 && (
        <p>No images found for single mode.</p>
      )}

      {/* ========== MULTIPLE MODE ========== */}
      {mode === "multiple" && (
        <div>
          <div style={{ marginBottom: "10px" }}>
            <label>Images per page:</label>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
              style={{ marginLeft: "5px" }}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          {images.length > 0 ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)", // 3 images per row
                gap: "10px",
                justifyItems: "center", // center each grid cell
                alignItems: "start",
              }}
            >
              {images.map((img, idx) => (
                <MultipleThumb
                  key={img.id}
                  image={img}
                  classId={currentClassId}
                  baseUrl={BASE_URL} // pass the baseUrl to subcomponent if needed
                  onDiscard={() => handleDiscard(img)}
                  onUpdateImage={(updated) => {
                    setImages((prev) => {
                      const copy = [...prev];
                      copy[idx] = updated;
                      return copy;
                    });
                  }}
                />
              ))}
            </div>
          ) : (
            <p>No images found for multiple mode.</p>
          )}

          <div style={{ marginTop: "20px" }}>
            <button onClick={() => setPage((p) => (p > 1 ? p - 1 : p))} disabled={page <= 1}>
              Previous
            </button>
            <span style={{ margin: "0 10px" }}>Page {page}</span>
            <button onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </div>
      )}
    </div>
  );
}

function MultipleThumb({ image, classId, onDiscard, onUpdateImage, baseUrl }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPt, setStartPt] = useState(null);

  useEffect(() => {
    drawBoxes();
  }, [image.boxes]);

  const drawBoxes = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // "virtual" 160x160 coordinate system
    ctx.clearRect(0, 0, 160, 160);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    image.boxes.forEach((b) => {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      // label
      const label = b.classId >= 0 ? b.classId : "?";
      ctx.fillStyle = "lime";
      ctx.font = "12px Arial";
      ctx.fillText(label, b.x + 2, b.y + 12);
    });
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    setStartPt({ x: sx, y: sy });
  };
  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    drawBoxes();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    ctx.strokeRect(startPt.x, startPt.y, mx - startPt.x, my - startPt.y);
  };
  const handleMouseUp = (e) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = mx - startPt.x;
    const h = my - startPt.y;
    const newBox = {
      classId,
      x: startPt.x,
      y: startPt.y,
      w,
      h,
    };
    const updated = { ...image, boxes: [...image.boxes, newBox] };
    onUpdateImage(updated);
    setStartPt(null);
  };

  const handleSave = async () => {
    try {
      const body = {
        boxes: image.boxes,
        imageWidth: 160, // multiple mode
        imageHeight: 160,
      };
      // We can read from baseUrl if needed, or do a direct reference again:
      const annUrl = `${BASE_URL}/api/annotations/${image.filename}`;
      const res = await fetch(annUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Save error " + res.status);
      alert("Saved for " + image.filename);
    } catch (err) {
      console.error("Save error:", err);
      alert("Error saving annotation");
    }
  };

  return (
    <div
      style={{
        // We'll fix a container for each thumbnail at ~25% scale
        position: "relative",
        width: "25%", // each item ~ 25% wide. Our grid has 3 items per row.
      }}
    >
      <div style={{ position: "relative", width: "100%", margin: "auto" }}>
        {/* The image scaled to ~25% in multiple mode */}
        <img
          src={image.dataUrl}
          alt={image.filename}
          style={{
            display: "block",
            width: "100%", // fill container
            height: "auto",
            objectFit: "cover",
          }}
        />
        {/* A "virtual" 160x160 coordinate system */}
        <canvas
          ref={canvasRef}
          width={160}
          height={160}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%", // match the scaled image
            height: "auto",
            cursor: "crosshair",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        />
      </div>
      {/* Save + Discard */}
      <div style={{ marginTop: "5px" }}>
        <button onClick={handleSave} style={{ fontSize: "12px", marginRight: "5px" }}>
          Save
        </button>
        {image.status === "discarded" ? (
          <button
            style={{ backgroundColor: "green", color: "white", fontSize: "12px" }}
          >
            Discarded
          </button>
        ) : (
          <button
            style={{ backgroundColor: "red", color: "white", fontSize: "12px" }}
            onClick={onDiscard}
          >
            Discard
          </button>
        )}
      </div>
    </div>
  );
}

export default ImageFilter;
