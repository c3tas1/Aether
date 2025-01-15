import React, { useState, useEffect } from "react";

function ImageFilter() {
  // =================== UPLOAD STATE ===================
  const [selectedFiles, setSelectedFiles] = useState([]);

  // =================== SEARCH / DISPLAY STATE ===================
  const [mode, setMode] = useState("single");
  const [searchQuery, setSearchQuery] = useState("");
  const [feeOption, setFeeOption] = useState("");

  // Each image: { id, filename, dataUrl, status }
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // Pagination states (multiple mode)
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // =================== UPLOAD ===================
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

  // =================== FETCH IMAGES ===================
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

      // Data is an array of objects: [{ id, filename, base64, status }, ...]
      const data = await res.json();
      console.log("Data from backend:", data);

      // Convert each to { id, filename, dataUrl, status }
      if (Array.isArray(data)) {
        const mapped = data.map((item) => ({
          id: item.id,
          filename: item.filename,
          dataUrl: `data:image/jpeg;base64,${item.base64}`,
          status: item.status || "", // might be "" or "discarded"
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

  // =================== DISCARD ===================
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

      // Update local state so the button turns green immediately
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
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchImages();
  };

  // =================== EFFECTS ===================
  useEffect(() => {
    if (searchQuery || feeOption) {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [mode]);

  useEffect(() => {
    if (mode === "multiple" && (searchQuery || feeOption)) {
      fetchImages();
    }
    // eslint-disable-next-line
  }, [page, pageSize]);

  // =================== SINGLE MODE NAV ===================
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

  // =================== PAGINATION ===================
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
      <h1>Image Viewer + Uploader</h1>

      {/* ====== UPLOAD FORM ====== */}
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

      {/* ====== SEARCH FORM ====== */}
      <form onSubmit={handleSearchSubmit} style={{ marginBottom: "20px" }}>
        <label htmlFor="searchQuery">Search:</label>
        <input
          id="searchQuery"
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Type keywords..."
          style={{ marginLeft: "5px" }}
        />

        <label htmlFor="feeOption" style={{ marginLeft: "10px" }}>
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

      {/* ====== SINGLE MODE ====== */}
      {mode === "single" && images.length > 0 && (
        <div>
          <div
            style={{
              width: "400px",
              height: "400px",
              margin: "auto",
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
          {/* Show filename */}
          <div style={{ textAlign: "center", marginBottom: "10px" }}>
            <strong>{images[currentImageIndex].filename}</strong>
          </div>

          {/* Discard button => green if discarded, red if not */}
          <div style={{ textAlign: "center" }}>
            <button style={{ marginRight: "10px" }} onClick={handlePreviousImage}>
              Left
            </button>
            <button style={{ marginRight: "10px" }} onClick={handleNextImage}>
              Right
            </button>

            {images[currentImageIndex].status === "discarded" ? (
              <button
                style={{ backgroundColor: "green", color: "white" }}
                onClick={() =>
                  handleDiscard(images[currentImageIndex].id)
                }
              >
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
        </div>
      )}

      {mode === "single" && images.length === 0 && (
        <p>No images to display. Try searching or changing fee option.</p>
      )}

      {/* ====== MULTIPLE MODE ====== */}
      {mode === "multiple" && (
        <div>
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
                      overflow: "hidden",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                    }}
                  >
                    <div style={{ width: "100%", height: "120px" }}>
                      <img
                        src={imgObj.dataUrl}
                        alt={imgObj.filename}
                        style={{
                          maxWidth: "100%",
                          maxHeight: "100%",
                          display: "block",
                          margin: "0 auto",
                        }}
                      />
                    </div>
                    {/* filename */}
                    <div
                      style={{
                        padding: "5px",
                        fontSize: "0.9em",
                        textAlign: "center",
                      }}
                    >
                      {imgObj.filename}
                    </div>

                    {/* Discard Button => green if discarded, else red */}
                    <button
                      style={{
                        marginBottom: "5px",
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

export default ImageFilter;
