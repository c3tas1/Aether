import React, { useState, useEffect } from "react";

function App() {
  // =================== UPLOAD STATE (Optional) ===================
  const [selectedFiles, setSelectedFiles] = useState([]);

  // =================== SEARCH / DISPLAY STATE ===================
  const [mode, setMode] = useState("single");
  const [searchQuery, setSearchQuery] = useState("");
  const [feeOption, setFeeOption] = useState("");

  // Each main image object: { id, filename, dataUrl, status }
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // ** Reference image **
  const [referenceImage, setReferenceImage] = useState(null); // dataURL if found
  const [refImageError, setRefImageError] = useState(false);

  // Pagination (multiple mode)
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // =================== (Optional) UPLOAD HANDLERS ===================
  const handleFileChange = (e) => {
    setSelectedFiles([...e.target.files]);
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) {
      alert("No files selected!");
      return;
    }

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("images", file);
      });

      const res = await fetch("http://127.0.0.1:5000/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }

      const result = await res.json();
      console.log("Upload response:", result);
      alert("Upload successful!");
      setSelectedFiles([]);
    } catch (err) {
      console.error("Error uploading files:", err);
      alert("Error uploading files. See console for details.");
    }
  };

  // =================== REFERENCE IMAGE FETCH ===================
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
        // Probably 404 => reference not available
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

  // =================== MAIN IMAGES FETCH ===================
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
      console.log("Fetch images result:", data);

      if (Array.isArray(data)) {
        // data => [{ id, filename, base64, status }, ...]
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

  // =================== DISCARD HANDLER ===================
  const handleDiscard = async (imageId) => {
    try {
      const res = await fetch(
        `http://127.0.0.1:5000/api/images/${imageId}/discard`,
        {
          method: "PUT",
        }
      );
      if (!res.ok) {
        throw new Error(`Discard failed with status ${res.status}`);
      }
      const result = await res.json();
      console.log("Discard result:", result);

      // Update local state => set status = "discarded"
      setImages((prev) =>
        prev.map((img) =>
          img.id === imageId ? { ...img, status: "discarded" } : img
        )
      );
    } catch (err) {
      console.error("Error discarding image:", err);
      alert("Error discarding image. See console for details.");
    }
  };

  // =================== SEARCH ===================
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    setPage(1);

    // First fetch the reference image
    await fetchReferenceImage(searchQuery);

    // Then fetch the main images
    fetchImages();
  };

  // =================== EFFECTS ===================
  // If you change mode or pagination in multiple mode, re-fetch the images
  useEffect(() => {
    if ((searchQuery || feeOption) && mode === "multiple") {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode, page, pageSize]);

  // If you want immediate fetch on mode switch for single too
  useEffect(() => {
    if ((searchQuery || feeOption) && mode === "single") {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode]);

  // =================== SINGLE MODE NAVIGATION ===================
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

  // =================== RENDER HELPER: REFERENCE IMAGE ===================
  const renderReferenceImage = () => {
    if (referenceImage) {
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
      // If not found or error => "Reference image not available"
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
          Reference image not available
        </div>
      );
    }
  };

  // =================== RENDER ===================
  return (
    <div style={{ margin: "20px" }}>
      <h1>Image Viewer with Reference & Discard</h1>

      {/* ====== (Optional) UPLOAD FORM ====== */}
      <h2>Upload Images or ZIP File</h2>
      <form onSubmit={handleUploadSubmit} style={{ marginBottom: "30px" }}>
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          accept=".zip,image/*"
          style={{ marginRight: "10px" }}
        />
        <button type="submit">Upload</button>
      </form>

      {/* ====== SEARCH FORM ====== */}
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

        <label htmlFor="feeOption">Fee:</label>
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

      {/* ====== MODE TOGGLE ====== */}
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
        <div style={{ display: "flex", gap: "20px" }}>
          {/* Main single image on the left */}
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
                <div style={{ textAlign: "center", marginBottom: "10px" }}>
                  <button onClick={handlePreviousImage}>Left</button>
                  <button onClick={handleNextImage} style={{ marginLeft: "10px" }}>
                    Right
                  </button>
                </div>

                {/* Discard button => green if status='discarded', else red */}
                <div style={{ textAlign: "center" }}>
                  {images[currentImageIndex].status === "discarded" ? (
                    <button style={{ backgroundColor: "green", color: "white" }}>
                      Discarded
                    </button>
                  ) : (
                    <button
                      style={{ backgroundColor: "red", color: "white" }}
                      onClick={() =>
                        handleDiscard(images[currentImageIndex].id)
                      }
                    >
                      Discard
                    </button>
                  )}
                </div>
              </>
            ) : (
              <p>No images to display. Try searching or changing fee option.</p>
            )}
          </div>

          {/* Reference image on the right */}
          <div>
            <h3>Reference Image</h3>
            {renderReferenceImage()}
          </div>
        </div>
      )}

      {/* ==================== MULTIPLE MODE ==================== */}
      {mode === "multiple" && (
        <div>
          <h3>Reference Image</h3>
          {renderReferenceImage()}

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
              {images.map((imgObj) => {
                const isDiscarded = imgObj.status === "discarded";
                return (
                  <div
                    key={imgObj.id}
                    style={{
                      border: "1px solid #ccc",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                    }}
                  >
                    <div
                      style={{
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
                        alt={imgObj.filename}
                        style={{ maxWidth: "100%", maxHeight: "100%" }}
                      />
                    </div>
                    {/* Discard button => green if discarded, else red */}
                    <button
                      style={{
                        margin: "5px",
                        backgroundColor: isDiscarded ? "green" : "red",
                        color: "white",
                      }}
                      onClick={() => handleDiscard(imgObj.id)}
                    >
                      {isDiscarded ? "Discarded" : "Discard"}
                    </button>
                  </div>
                );
              })}
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
