import React, { useState, useEffect } from "react";

function App() {
  // =================== UPLOAD STATE (if you still have uploads) ===================
  const [selectedFiles, setSelectedFiles] = useState([]);

  // =================== SEARCH / DISPLAY STATE ===================
  const [mode, setMode] = useState("single");
  const [searchQuery, setSearchQuery] = useState("");
  const [feeOption, setFeeOption] = useState("");

  // Each main image object: { id, filename, dataUrl, status }
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // For reference image
  const [referenceImage, setReferenceImage] = useState(null); // store dataURL or null
  const [refImageError, setRefImageError] = useState(false);

  // Pagination (multiple mode)
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // =================== FETCH REFERENCE IMAGE ===================
  const fetchReferenceImage = async (searchTerm) => {
    try {
      if (!searchTerm) {
        setReferenceImage(null);
        setRefImageError(true);
        return;
      }
      const refRes = await fetch(
        `http://127.0.0.1:5000/api/reference_image?search=${encodeURIComponent(
          searchTerm
        )}`
      );
      if (!refRes.ok) {
        // probably 404
        setReferenceImage(null);
        setRefImageError(true);
        return;
      }
      const refData = await refRes.json();
      if (refData.base64) {
        const dataUrl = `data:image/jpeg;base64,${refData.base64}`;
        setReferenceImage(dataUrl);
        setRefImageError(false);
      } else {
        setReferenceImage(null);
        setRefImageError(true);
      }
    } catch (err) {
      console.error("Error fetching reference image:", err);
      setReferenceImage(null);
      setRefImageError(true);
    }
  };

  // =================== FETCH MAIN IMAGES ===================
  const fetchImages = async () => {
    try {
      let endpoint = "";
      if (mode === "single") {
        endpoint = `http://127.0.0.1:5000/api/images/single?search=${searchQuery}&fee=${feeOption}&page=${page}&per_page=${pageSize}`;
      } else {
        endpoint = `http://127.0.0.1:5000/api/images/multiple?search=${searchQuery}&fee=${feeOption}&page=${page}&limit=${pageSize}`;
      }

      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error(`Network response was not ok: ${res.status}`);
      }

      const data = await res.json();
      // data is an array of objects: [{ id, filename, base64, status }, ...]
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

      setCurrentImageIndex(0);
    } catch (err) {
      console.error("Error fetching images:", err);
    }
  };

  // =================== HANDLE SEARCH ===================
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    setPage(1);

    // 1) Fetch reference image for the search term
    await fetchReferenceImage(searchQuery);

    // 2) Then fetch main images
    fetchImages();
  };

  // =================== USE EFFECTS ===================
  // If you change mode, or page/pageSize in multiple mode, we re-fetch images (and keep the same reference)
  useEffect(() => {
    if ((searchQuery || feeOption) && mode === "multiple") {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode, page, pageSize]);

  // If you want immediate fetch on mode switch in single mode too, you could do:
  useEffect(() => {
    if ((searchQuery || feeOption) && mode === "single") {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode]);

  // =================== RENDERING REFERENCE IMAGE ===================
  const renderReferenceImage = () => {
    if (referenceImage) {
      // We have a dataURL
      return (
        <div
          style={{
            border: "1px solid #ccc",
            width: "200px",
            height: "200px",
            margin: "auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "10px",
          }}
        >
          <img
            src={referenceImage}
            alt="Reference"
            style={{ maxWidth: "100%", maxHeight: "100%" }}
          />
        </div>
      );
    } else {
      // If not found or error
      return (
        <div
          style={{
            border: "1px solid #ccc",
            width: "200px",
            height: "200px",
            margin: "auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "10px",
            color: "red",
          }}
        >
          <p>Reference image not available</p>
        </div>
      );
    }
  };

  // =================== SINGLE MODE NAV (unchanged) ===================
  const handlePreviousImage = () => {
    setCurrentImageIndex((prevIndex) =>
      prevIndex === 0 ? images.length - 1 : prevIndex - 1
    );
  };

  const handleNextImage = () => {
    setCurrentImageIndex((prevIndex) =>
      prevIndex === images.length - 1 ? 0 : prevIndex + 1
    );
  };

  // =================== MULTIPLE MODE PAGINATION ===================
  const handlePreviousPage = () => {
    if (page > 1) {
      setPage((prev) => prev - 1);
    }
  };

  const handleNextPage = () => {
    setPage((prev) => prev + 1);
  };

  const handlePageSizeChange = (e) => {
    setPageSize(parseInt(e.target.value, 10));
    setPage(1);
  };

  // =================== RENDER ===================
  return (
    <div style={{ margin: "20px" }}>
      <h1>Image Viewer + Reference Image</h1>

      {/* ========== SEARCH FORM ========== */}
      <form onSubmit={handleSearchSubmit} style={{ marginBottom: "20px" }}>
        <label htmlFor="searchQuery">Search:</label>
        <input
          id="searchQuery"
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Type keywords..."
          style={{ marginLeft: "5px", marginRight: "10px" }}
        />

        <label htmlFor="feeOption">
          Fee:
        </label>
        <select
          id="feeOption"
          value={feeOption}
          onChange={(e) => setFeeOption(e.target.value)}
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

      {/* ========== MODE TOGGLE ========== */}
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

      {/* ==================== SINGLE MODE ==================== */}
      {mode === "single" && (
        <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
          {/* Left or main content: single image */}
          <div>
            {images.length > 0 ? (
              <>
                <div
                  style={{
                    width: "400px",
                    height: "400px",
                    border: "1px solid #ccc",
                    marginBottom: "10px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <img
                    src={images[currentImageIndex].dataUrl}
                    alt="Single mode"
                    style={{ maxWidth: "100%", maxHeight: "100%" }}
                  />
                </div>
                <div style={{ textAlign: "center" }}>
                  <button onClick={handlePreviousImage}>Left</button>
                  <button
                    onClick={handleNextImage}
                    style={{ marginLeft: "10px" }}
                  >
                    Right
                  </button>
                </div>
              </>
            ) : (
              <p>No images to display. Try searching or changing fee option.</p>
            )}
          </div>

          {/* Right side: reference image */}
          <div>
            <h3>Reference Image</h3>
            {renderReferenceImage()}
          </div>
        </div>
      )}

      {/* ==================== MULTIPLE MODE ==================== */}
      {mode === "multiple" && (
        <div>
          {/* Reference image on top row */}
          <h3>Reference Image</h3>
          <div style={{ marginBottom: "20px" }}>
            {renderReferenceImage()}
          </div>

          {/* PAGE SIZE SELECTOR */}
          <div style={{ marginBottom: "10px" }}>
            <label htmlFor="pageSize">Images per page:</label>
            <select
              id="pageSize"
              value={pageSize}
              onChange={handlePageSizeChange}
              style={{ marginLeft: "5px" }}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={250}>250</option>
            </select>
          </div>

          {/* IMAGE GRID */}
          {images.length > 0 ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(5, 1fr)",
                gap: "10px",
              }}
            >
              {images.map((imgObj, idx) => (
                <div
                  key={idx}
                  style={{
                    border: "1px solid #ccc",
                    width: "100%",
                    height: "120px",
                    overflow: "hidden",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                >
                  <img
                    src={imgObj.dataUrl}
                    alt={`Grid item ${idx}`}
                    style={{ maxWidth: "100%", maxHeight: "100%" }}
                  />
                </div>
              ))}
            </div>
          ) : (
            <p>No images found. Try adjusting search or fee filter.</p>
          )}

          {/* PAGINATION */}
          <div style={{ marginTop: "20px", textAlign: "center" }}>
            <button onClick={handlePreviousPage} disabled={page === 1}>
              Previous Page
            </button>
            <span style={{ margin: "0 10px" }}>Page {page}</span>
            <button onClick={handleNextPage}>Next Page</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
