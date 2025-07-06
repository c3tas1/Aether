import React, { useState, useEffect, useRef, useCallback } from "react";
import Select from 'react-select';
import './ImageFilter.css';

// Constants
const BASE_URL = "http://127.0.0.1:5000";
const SINGLE_LOGICAL_WIDTH = 1200;
const SINGLE_LOGICAL_HEIGHT = 1600;
const MULTIPLE_LOGICAL_WIDTH = 160;
const MULTIPLE_LOGICAL_HEIGHT = 160;
const CLASS_NAMES = ["person", "car", "dog", "cat", "bottle", "chair", "tv", "phone", "book", "cup", "laptop", "boat", "bird", "plane", "cow", "sheep"];
const MODEL_OPTIONS = [{ value: 'yolov8n', label: 'YOLOv8n' }, { value: 'yolov8s', label: 'YOLOv8s' }, { value: 'yolov8m', label: 'YOLOv8m' }];

function ImageFilter() {
    // --- State Hooks ---
    const [datasetId, setDatasetId] = useState(null);
    const [imageManifest, setImageManifest] = useState([]); // Holds the list of all file paths from the manifest
    const [images, setImages] = useState([]); // Holds only the currently loaded image data
    const [isLoading, setIsLoading] = useState(true); // For showing loading indicators

    const [searchQuery, setSearchQuery] = useState("");
    const [feeOption, setFeeOption] = useState("");
    const [mode, setMode] = useState("single");
    const [currentIndex, setCurrentIndex] = useState(0);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(25);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);
    const [currentClassId, setCurrentClassId] = useState(0);
    const [selectedModel, setSelectedModel] = useState(null);
    const singleCanvasRef = useRef(null);

    // --- Effect 1: Fetch Manifest on Initial Load ---
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const dataset = params.get('dataset');
        if (dataset) {
            setDatasetId(dataset);
            fetchManifest(dataset);
        } else {
            setIsLoading(false); // No dataset to load, so stop loading indicator
        }
    }, []);

    const fetchManifest = async (datasetId) => {
        setIsLoading(true);
        try {
            const manifestRes = await fetch(`${BASE_URL}/api/datasets/${datasetId}/preview?path=manifest.json`);
            if (!manifestRes.ok) throw new Error(`Could not fetch dataset manifest. Status: ${manifestRes.status}`);
            
            const manifestData = await manifestRes.json();
            const manifest = JSON.parse(manifestData.content);
            setImageManifest(manifest); // Store the list of all image file paths
        } catch (err) {
            console.error("Fetch manifest error:", err);
            alert(`Failed to load dataset manifest: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    // --- Effect 2: Load Image Data On-Demand ---
    useEffect(() => {
        if (imageManifest.length === 0 || !datasetId) return;

        const loadImagesForView = async () => {
            setIsLoading(true);
            let pathsToLoad = [];

            if (mode === 'single') {
                if (imageManifest[currentIndex]) {
                    pathsToLoad = [imageManifest[currentIndex].image];
                }
            } else {
                const startIndex = (page - 1) * pageSize;
                const endIndex = startIndex + pageSize;
                pathsToLoad = imageManifest.slice(startIndex, endIndex).map(item => item.image);
            }

            if (pathsToLoad.length === 0) {
                setImages([]);
                setIsLoading(false);
                return;
            }

            try {
                const batchRes = await fetch(`${BASE_URL}/api/datasets/${datasetId}/image-batch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(pathsToLoad),
                });
                if (!batchRes.ok) throw new Error(`Failed to fetch image batch. Status: ${batchRes.status}`);
                
                const loadedImagesData = await batchRes.json();
                setImages(loadedImagesData);
            } catch (err) {
                console.error("Image loading error:", err);
                alert(err.message);
                setImages([]);
            } finally {
                setIsLoading(false);
            }
        };

        loadImagesForView();
    }, [currentIndex, page, pageSize, mode, imageManifest, datasetId]);

    // --- Drawing and Annotation Functions ---
    const drawSingleBoxes = useCallback(() => {
        if (mode !== "single" || !singleCanvasRef.current || images.length === 0) return;
        const canvas = singleCanvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, SINGLE_LOGICAL_WIDTH, SINGLE_LOGICAL_HEIGHT);
        
        // In single mode, 'images' array will only have one item
        const img = images[0]; 
        if (!img) return;

        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.font = "18px 'Fira Code', monospace";
        if(img.boxes) {
            img.boxes.forEach((box) => {
                ctx.strokeRect(box.x, box.y, box.w, box.h);
                const label = CLASS_NAMES[box.classId] || `id=${box.classId}`;
                ctx.fillStyle = "#00FF00";
                ctx.fillText(label, box.x + 2, box.y + 18);
            });
        }
    }, [mode, images]);

    useEffect(() => {
        if (mode === "single") {
            drawSingleBoxes();
        }
    }, [images, mode, drawSingleBoxes]);

    const handleSingleMouseDown = (e) => {
        setIsDrawing(true);
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = SINGLE_LOGICAL_WIDTH / rect.width;
        const scaleY = SINGLE_LOGICAL_HEIGHT / rect.height;
        const sx = (e.clientX - rect.left) * scaleX;
        const sy = (e.clientY - rect.top) * scaleY;
        setStartPt({ x: sx, y: sy });
    };

    const handleSingleMouseMove = (e) => {
        if (!isDrawing || !startPt) return;
        drawSingleBoxes();
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = SINGLE_LOGICAL_WIDTH / rect.width;
        const scaleY = SINGLE_LOGICAL_HEIGHT / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const canvas = singleCanvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 2;
        ctx.strokeRect(startPt.x, startPt.y, mx - startPt.x, my - startPt.y);
    };

    const handleSingleMouseUp = (e) => {
        if (!isDrawing || !startPt) return;
        setIsDrawing(false);
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = SINGLE_LOGICAL_WIDTH / rect.width;
        const scaleY = SINGLE_LOGICAL_HEIGHT / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const w = mx - startPt.x;
        const h = my - startPt.y;
        if (Math.abs(w) < 1 || Math.abs(h) < 1) return;
        
        const newBox = { classId: currentClassId, x: startPt.x, y: startPt.y, w, h };
        
        // Update the state correctly for single-image view
        const updatedImages = [...images];
        if (!updatedImages[0].boxes) updatedImages[0].boxes = [];
        updatedImages[0].boxes.push(newBox);
        setImages(updatedImages);
        setStartPt(null);
    };

    const saveSingleAnnotations = async () => {
        if (images.length === 0) return;
        const img = images[0];
        try {
            const body = { boxes: img.boxes, imageWidth: SINGLE_LOGICAL_WIDTH, imageHeight: SINGLE_LOGICAL_HEIGHT };
            const res = await fetch(`${BASE_URL}/api/images/${img.id}/annotations`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error("Save error " + res.status);
            alert("Saved annotations for " + img.filename);
        } catch (err) {
            console.error("saveSingleAnnotations error:", err);
            alert("Save failed. See console.");
        }
    };

    const handleDiscard = async (img) => {
        // This function might need backend adjustments to remove an image from the manifest
        console.log("Discarding image:", img.id);
        alert("Discard functionality placeholder.");
    };
    
    const handleSearchSubmit = (e) => {
        e.preventDefault();
        // This would need a separate fetch implementation if used without a datasetId
        alert("Search is disabled when viewing a specific dataset.");
    };

    const handlePrevImage = () => {
        setCurrentIndex((i) => (i === 0 ? imageManifest.length - 1 : i - 1));
    };

    const handleNextImage = () => {
        setCurrentIndex((i) => (i === imageManifest.length - 1 ? 0 : i + 1));
    };

    // --- Render Logic ---
    return (
        <div className="image-filter-container">
            <h1 className="page-title">{'// ANNOTATION_INTERFACE'}</h1>
            {datasetId && <h2 className="dataset-subtitle">Annotating Dataset: {datasetId}</h2>}
            
            <div className="controls-grid">
                <fieldset className="control-group filter-search-panel" disabled={!!datasetId}>
                    <legend>FILTER & SEARCH</legend>
                    <form onSubmit={handleSearchSubmit}>
                        <input className="terminal-input" placeholder="Search disabled" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} disabled={!!datasetId} />
                        <select className="terminal-select" value={feeOption} onChange={(e) => setFeeOption(e.target.value)} disabled={!!datasetId}>
                            <option value="">FEE: ALL</option>
                        </select>
                        <button type="submit" className="terminal-button primary" disabled={!!datasetId}>Search</button>
                    </form>
                </fieldset>
                <fieldset className="control-group view-mode-panel">
                    <legend>VIEW MODE</legend>
                    <div className="mode-toggle-group">
                        <button onClick={() => setMode("single")} className={`terminal-button ${mode === 'single' ? 'active' : ''}`}>Single Mode</button>
                        <button onClick={() => setMode("multiple")} className={`terminal-button ${mode === 'multiple' ? 'active' : ''}`}>Multiple Mode</button>
                    </div>
                </fieldset>
                <fieldset className="control-group models-panel">
                    <legend>MODELS</legend>
                    <Select className="react-select-container" classNamePrefix="react-select" options={MODEL_OPTIONS} value={selectedModel} onChange={setSelectedModel} placeholder="Select a model..." isSearchable />
                </fieldset>
            </div>
            <fieldset className="class-selector-panel">
                <legend>ANNOTATION CLASS</legend>
                <div className="class-buttons-container">
                    {CLASS_NAMES.map((name, idx) => (
                        <button key={idx} className={`terminal-button ${currentClassId === idx ? 'active' : ''}`} onClick={() => setCurrentClassId(idx)}>{name}</button>
                    ))}
                </div>
            </fieldset>

            <div className="main-content-area">
                {isLoading ? (
                    <div className="spinner-container"><div className="spinner"></div></div>
                ) : (
                    <>
                        {mode === "single" && images.length > 0 && imageManifest.length > 0 && (
                            <div className="single-mode-container">
                                <div className="image-info">
                                    Image {currentIndex + 1} of {imageManifest.length}
                                </div>
                                <div className="single-image-wrapper">
                                    <img src={`data:image/jpeg;base64,${images[0].base64}`} alt="Annotation target" className="main-image" />
                                    <canvas ref={singleCanvasRef} width={SINGLE_LOGICAL_WIDTH} height={SINGLE_LOGICAL_HEIGHT} className="annotation-canvas" onMouseDown={handleSingleMouseDown} onMouseMove={handleSingleMouseMove} onMouseUp={handleSingleMouseUp} />
                                </div>
                                <div className="action-bar">
                                    <button className="terminal-button" onClick={saveSingleAnnotations}>Save Annotations</button>
                                    <button className="terminal-button discard" onClick={() => handleDiscard(images[0])}>{images[0].status === "discarded" ? "Discarded" : "Discard"}</button>
                                    <button className="terminal-button" onClick={handlePrevImage}>Prev</button>
                                    <button className="terminal-button" onClick={handleNextImage}>Next</button>
                                </div>
                            </div>
                        )}
                        {mode === "multiple" && imageManifest.length > 0 && (
                            <div className="multiple-mode-container">
                                <div className="multiple-mode-grid">
                                    {images.map((img) => (
                                        <MultipleThumb key={img.id} image={img} classId={currentClassId} onDiscard={() => handleDiscard(img)} onUpdateImage={() => {}} />
                                    ))}
                                </div>
                                <div className="pagination-controls">
                                    <button className="terminal-button" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1}>Previous</button>
                                    <span>Page {page} of {Math.ceil(imageManifest.length / pageSize)}</span>
                                    <button className="terminal-button" onClick={() => setPage((p) => p + 1)} disabled={page * pageSize >= imageManifest.length}>Next</button>
                                </div>
                            </div>
                        )}
                        {!isLoading && imageManifest.length === 0 && (
                             <p className="status-text">{datasetId ? "No images found in this dataset." : "No dataset loaded."}</p>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

function MultipleThumb({ image, classId, onDiscard, onUpdateImage }) {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);

    const drawBoxes = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, MULTIPLE_LOGICAL_WIDTH, MULTIPLE_LOGICAL_HEIGHT);
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 1;
        if(image.boxes) {
            image.boxes.forEach((b) => {
                ctx.strokeRect(b.x, b.y, b.w, b.h);
                const label = CLASS_NAMES[b.classId] || `?`;
                ctx.fillStyle = "#00FF00";
                ctx.font = "12px 'Fira Code', monospace";
                ctx.fillText(label, b.x + 2, b.y + 12);
            });
        }
    }, [image.boxes]);

    useEffect(() => { drawBoxes(); }, [drawBoxes]);

    const handleMouseDown = (e) => {
        setIsDrawing(true);
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = MULTIPLE_LOGICAL_WIDTH / rect.width;
        const scaleY = MULTIPLE_LOGICAL_HEIGHT / rect.height;
        const sx = (e.clientX - rect.left) * scaleX;
        const sy = (e.clientY - rect.top) * scaleY;
        setStartPt({ x: sx, y: sy });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing) return;
        drawBoxes();
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = MULTIPLE_LOGICAL_WIDTH / rect.width;
        const scaleY = MULTIPLE_LOGICAL_HEIGHT / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const ctx = canvasRef.current.getContext("2d");
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 1;
        ctx.strokeRect(startPt.x, startPt.y, mx - startPt.x, my - startPt.y);
    };

    const handleMouseUp = (e) => {
        if (!isDrawing) return;
        setIsDrawing(false);
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = MULTIPLE_LOGICAL_WIDTH / rect.width;
        const scaleY = MULTIPLE_LOGICAL_HEIGHT / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const w = mx - startPt.x;
        const h = my - startPt.y;
        const newBox = { classId, x: startPt.x, y: startPt.y, w, h };
        const updated = { ...image, boxes: [...(image.boxes || []), newBox] };
        onUpdateImage(updated);
        setStartPt(null);
    };

    const handleSave = async () => {
        try {
            const body = { boxes: image.boxes, imageWidth: MULTIPLE_LOGICAL_WIDTH, imageHeight: MULTIPLE_LOGICAL_HEIGHT };
            const res = await fetch(`${BASE_URL}/api/images/${image.id}/annotations`, {
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
        <div className="thumb-card">
            <div className="thumb-image-wrapper">
                <img src={`data:image/jpeg;base64,${image.base64}`} alt={image.filename} className="main-image" />
                <canvas ref={canvasRef} width={MULTIPLE_LOGICAL_WIDTH} height={MULTIPLE_LOGICAL_HEIGHT} className="annotation-canvas" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} />
            </div>
            <div className="thumb-controls">
                <button className="terminal-button small" onClick={handleSave}>Save</button>
                <button className={`terminal-button small ${image.status === "discarded" ? 'discarded' : 'discard'}`} onClick={onDiscard} disabled={image.status === "discarded"}>
                    {image.status === "discarded" ? 'Discarded' : 'Discard'}
                </button>
            </div>
        </div>
    );
}

export default ImageFilter;
