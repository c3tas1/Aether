import React, { useState } from 'react';

function App() {
  const [mode, setMode] = useState("single");
  const [searchQuery, setSearchQuery] = useState("");
  const [feeOption, setFeeOption] = useState("");
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const fetchImages = async () => {
    try {
      let endpoint = "";
      if (mode === "single") {
        endpoint = `/api/images/single?search=${searchQuery}&fee=${feeOption}`;
      } else {
        endpoint = `/api/images/multiple?search=${searchQuery}&fee=${feeOption}&page=${page}&limit=${pageSize}`;
      }

      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error(`Network response was not ok: ${res.status}`);
      }

      const data = await res.json();
      if (data.images) {
        setImages(data.images);
      } else if (Array.isArray(data)) {
        setImages(data);
      } else {
        setImages([]);
      }

      // Reset if single mode
      setCurrentImageIndex(0);
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  // Only called on "Search" form submit
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchImages();
  };

  // Example: Next/Prev page calls fetchImages only upon explicit user action
  const handleNextPage = () => {
    setPage((prev) => prev + 1);
    fetchImages();
  };
  const handlePreviousPage = () => {
    if (page > 1) {
      setPage((prev) => prev - 1);
      fetchImages();
    }
  };
  const handlePageSizeChange = (e) => {
    setPageSize(Number(e.target.value));
    setPage(1);
    fetchImages();
  };

  return (
    <div>
      <div>
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

      <form onSubmit={handleSearchSubmit}>
        <label>Search: </label>
        <input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <label>Fee: </label>
        <select
          value={feeOption}
          onChange={(e) => setFeeOption(e.target.value)}
        >
          <option value="">All</option>
          <option value="free">Free</option>
          <option value="paid">Paid</option>
        </select>
        <button type="submit">Search</button>
      </form>

      {mode === "single" ? (
        <div>
          {images.length > 0 ? (
            <>
              <img src={images[currentImageIndex]} alt="Single" />
              <button onClick={() =>
                setCurrentImageIndex((prev) =>
                  prev === 0 ? images.length - 1 : prev - 1
                )
              }>
                Left
              </button>
              <button onClick={() =>
                setCurrentImageIndex((prev) =>
                  prev === images.length - 1 ? 0 : prev + 1
                )
              }>
                Right
              </button>
            </>
          ) : (
            <p>No images</p>
          )}
        </div>
      ) : (
        <div>
          <label>Page Size:</label>
          <select value={pageSize} onChange={handlePageSizeChange}>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
          <button onClick={handlePreviousPage} disabled={page <= 1}>
            Prev
          </button>
          <span>{page}</span>
          <button onClick={handleNextPage}>Next</button>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)" }}>
            {images.map((img, i) => (
              <img key={i} src={img} alt={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
