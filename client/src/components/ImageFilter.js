import React, { useState, useEffect, useRef, useCallback } from "react";

// --- CONFIGURATION & CONSTANTS ---
const BASE_URL = "http://127.0.0.1:5000";
const SINGLE_MODE_CANVAS_WIDTH = 1400; 
const SINGLE_MODE_CANVAS_HEIGHT = 1400;
const MULTIPLE_MODE_CANVAS_WIDTH = 700;
const MULTIPLE_MODE_CANVAS_HEIGHT = 700;
const CLASS_COLORS = [ 
    "#E53935", "#1E88E5", "#43A047", "#FDD835", "#8E24AA", "#FB8C00",
    "#00ACC1", "#D81B60", "#5E35B1", "#039BE5", "#7CB342", "#6D4C41",
    "#546E7A", "#F4511E", "#00897B", "#3949AB"
];
const ANNOTATION_LINE_WIDTH = 2;
const ANNOTATION_FILL_ALPHA = 0.2;
const ANNOTATION_FONT_SIZE = 16;
const ANNOTATION_THUMB_FONT_SIZE = 12;
const TEMP_ANNOTATION_COLOR = "rgba(173, 216, 230, 0.7)";
const AUTOSAVE_DELAY = 1500; // 1.5 seconds

// --- GLOBAL HELPER: Draw Bounding Box ---
const drawBox = (ctx, box, offsetX, offsetY, scaleX, scaleY, isTemp, isThumb, classNamesList) => {
    const color = CLASS_COLORS[box.classId % CLASS_COLORS.length];
    const label = (classNamesList && classNamesList[box.classId] !== undefined) ? classNamesList[box.classId] : `ID: ${box.classId}`;
    const scaledX = offsetX + box.x * scaleX;
    const scaledY = offsetY + box.y * scaleY;
    const scaledW = box.w * scaleX;
    const scaledH = box.h * scaleY;
    ctx.strokeStyle = isTemp ? TEMP_ANNOTATION_COLOR : color;
    ctx.lineWidth = isTemp ? 2 : (isThumb ? 1.5 : ANNOTATION_LINE_WIDTH);
    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
    if (!isTemp) {
        ctx.fillStyle = `${color}${Math.round(ANNOTATION_FILL_ALPHA * 255).toString(16).padStart(2, '0')}`;
        ctx.fillRect(scaledX, scaledY, scaledW, scaledH);
        
        if (label && ((scaledW > 30 && scaledH > 20) || !isThumb)) {
            const fontSize = isThumb ? ANNOTATION_THUMB_FONT_SIZE : ANNOTATION_FONT_SIZE;
            ctx.font = `bold ${fontSize}px Arial`;
            const textMetrics = ctx.measureText(label);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.fillRect(scaledX, scaledY, textMetrics.width + 8, fontSize + 4);
            
            ctx.fillStyle = "#000000";
            ctx.fillText(label, scaledX + 4, scaledY + fontSize);
        }

        if (!isThumb) {
            ctx.fillStyle = "red";
            ctx.fillRect(scaledX + scaledW - 15, scaledY, 15, 15);
            ctx.fillStyle = "white";
            ctx.font = "bold 14px Arial";
            ctx.fillText("X", scaledX + scaledW - 12, scaledY + 12);
        }
    }
};

// --- Child Component: Classes Panel ---
function ClassesPanel({ classNames, currentClassId, onClassSelect }) {
    if (!classNames || classNames.length === 0) {
        return (
            <div style={styles.classesPanel.container}>
                <h3 style={styles.classesPanel.title}>Classes</h3>
                <p style={styles.classesPanel.placeholder}>Select a dataset to see classes.</p>
            </div>
        );
    }
    return (
        <div style={styles.classesPanel.container}>
            <h3 style={styles.classesPanel.title}>Classes</h3>
            <div style={styles.classesPanel.list}>
                {classNames.map((name, idx) => (
                    <button
                        key={idx}
                        onClick={() => onClassSelect(idx)}
                        style={idx === currentClassId ? styles.classesPanel.buttonActive : styles.classesPanel.button}
                    >
                        {name}
                    </button>
                ))}
            </div>
        </div>
    );
}


// --- Child Component: Multiple View Thumbnail ---
function MultipleThumb({ image, classId, onUpdateImage, currentClassNames }) {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);
    const imgCacheRefThumb = useRef(new Image());

    const drawMultipleCanvasContent = useCallback((ctx, img, tempBox = null) => {
        ctx.clearRect(0, 0, MULTIPLE_MODE_CANVAS_WIDTH, MULTIPLE_MODE_CANVAS_HEIGHT);
        const { original_width: originalWidth, original_height: originalHeight } = img;
        if (!originalWidth || !originalHeight) return;
        const aspectRatio = originalWidth / originalHeight;
        const drawWidth = aspectRatio > 1 ? MULTIPLE_MODE_CANVAS_WIDTH : MULTIPLE_MODE_CANVAS_HEIGHT * aspectRatio;
        const drawHeight = aspectRatio > 1 ? MULTIPLE_MODE_CANVAS_WIDTH / aspectRatio : MULTIPLE_MODE_CANVAS_HEIGHT;
        const offsetX = (MULTIPLE_MODE_CANVAS_WIDTH - drawWidth) / 2;
        const offsetY = (MULTIPLE_MODE_CANVAS_HEIGHT - drawHeight) / 2;
        const scaleX = drawWidth / originalWidth;
        const scaleY = drawHeight / originalHeight;
        if (imgCacheRefThumb.current.src !== img.dataUrl || !imgCacheRefThumb.current.complete) {
            imgCacheRefThumb.current.src = img.dataUrl;
            imgCacheRefThumb.current.onload = () => {
                ctx.drawImage(imgCacheRefThumb.current, offsetX, offsetY, drawWidth, drawHeight);
                img.boxes.forEach(box => drawBox(ctx, box, offsetX, offsetY, scaleX, scaleY, false, true, currentClassNames));
                if (tempBox) drawBox(ctx, tempBox, offsetX, offsetY, scaleX, scaleY, true, true, currentClassNames);
            };
        } else {
            ctx.drawImage(imgCacheRefThumb.current, offsetX, offsetY, drawWidth, drawHeight);
            img.boxes.forEach(box => drawBox(ctx, box, offsetX, offsetY, scaleX, scaleY, false, true, currentClassNames));
            if (tempBox) drawBox(ctx, tempBox, offsetX, offsetY, scaleX, scaleY, true, true, currentClassNames);
        }
    }, [currentClassNames]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !image.dataUrl) return;
        const ctx = canvas.getContext("2d");
        drawMultipleCanvasContent(ctx, image);
    }, [image, drawMultipleCanvasContent]);

    const getOriginalCoords = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const { original_width: originalWidth, original_height: originalHeight } = image;
        if (!originalWidth || !originalHeight) return null;
        const aspectRatio = originalWidth / originalHeight;
        const drawWidth = aspectRatio > 1 ? MULTIPLE_MODE_CANVAS_WIDTH : MULTIPLE_MODE_CANVAS_HEIGHT * aspectRatio;
        const drawHeight = aspectRatio > 1 ? MULTIPLE_MODE_CANVAS_WIDTH / aspectRatio : MULTIPLE_MODE_CANVAS_HEIGHT;
        const offsetX = (MULTIPLE_MODE_CANVAS_WIDTH - drawWidth) / 2;
        const offsetY = (MULTIPLE_MODE_CANVAS_HEIGHT - drawHeight) / 2;
        const canvasX = (e.clientX - rect.left) * (MULTIPLE_MODE_CANVAS_WIDTH / rect.width);
        const canvasY = (e.clientY - rect.top) * (MULTIPLE_MODE_CANVAS_HEIGHT / rect.height);
        return {
            x: (canvasX - offsetX) * (originalWidth / drawWidth),
            y: (canvasY - offsetY) * (originalHeight / drawHeight),
        };
    };

    const handleMouseDown = (e) => { setIsDrawing(true); setStartPt(getOriginalCoords(e)); };

    const handleMouseMove = (e) => {
        if (!isDrawing || !startPt) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const currentPt = getOriginalCoords(e);
        if (!currentPt) return;
        const tempBox = { x: startPt.x, y: startPt.y, w: currentPt.x - startPt.x, h: currentPt.y - startPt.y, classId: classId };
        drawMultipleCanvasContent(ctx, image, tempBox);
    };

    const handleMouseUp = (e) => {
        if (!isDrawing || !startPt) return;
        setIsDrawing(false);
        const endPt = getOriginalCoords(e);
        if (!endPt) { setStartPt(null); return; }
        let [x1, y1, x2, y2] = [startPt.x, startPt.y, endPt.x, endPt.y];
        const newBox = { classId, x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x1 - x2), h: Math.abs(y1 - y2) };
        if (newBox.w > 2 && newBox.h > 2) {
            onUpdateImage({ ...image, boxes: [...image.boxes, newBox] });
        }
        setStartPt(null);
    };

    return (
        <div style={styles.thumb.card}>
            <div style={styles.thumb.canvasContainer}>
                <canvas ref={canvasRef} width={MULTIPLE_MODE_CANVAS_WIDTH} height={MULTIPLE_MODE_CANVAS_HEIGHT} style={styles.thumb.canvas} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} />
            </div>
            <div style={styles.thumb.info}>
                <p style={styles.thumb.filename}>{image.original_name}</p>
            </div>
        </div>
    );
}

// --- Main ImageFilter Component ---
function ImageFilter() {
    // --- STATE MANAGEMENT ---
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [datasetName, setDatasetName] = useState("");
    const [searchQuery, setSearchQuery] = useState("");
    const [feeOption, setFeeOption] = useState("Obj Det");
    const [mode, setMode] = useState("single");
    const [images, setImages] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [searchDataset, setSearchDataset] = useState("");
    const [searchClass, setSearchClass] = useState("");
    const [currentClassNames, setCurrentClassNames] = useState([]);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(10);
    const [totalImagesCount, setTotalImagesCount] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);
    const [currentClassId, setCurrentClassId] = useState(0);
    const [history, setHistory] = useState([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [autosaveStatus, setAutosaveStatus] = useState("Saved");
    const singleCanvasRef = useRef(null);
    const imgCacheRef = useRef(new Image());
    const autosaveTimeoutRef = useRef(null);

    // --- DATA FETCHING & API CALLS ---
    const handleUploadSubmit = async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0 || !datasetName.trim()) {
            alert("Please select files and provide a dataset name.");
            return;
        }
        setIsLoading(true);
        const formData = new FormData();
        selectedFiles.forEach(file => formData.append("images", file));
        formData.append("datasetName", datasetName);
        try {
            const res = await fetch(`${BASE_URL}/api/upload`, { method: "POST", body: formData });
            if (!res.ok) throw new Error(`Upload failed: ${await res.text()}`);
            const result = await res.json();
            alert("Upload successful!");
            setSelectedFiles([]);
            setDatasetName("");
            setSearchDataset(result.data[0]?.dataset_name || datasetName);
            handleSearchSubmit(e);
        } catch (err) {
            console.error("Upload error:", err);
            alert(err.message);
        } finally {
            setIsLoading(false);
        }
    };
    
    const fetchClasses = useCallback(async (dataset) => {
        if (!dataset) {
            setCurrentClassNames([]);
            return;
        }
        try {
            const res = await fetch(`${BASE_URL}/api/classes/${encodeURIComponent(dataset)}`);
            if (!res.ok) throw new Error("Could not fetch classes.");
            const classes = await res.json();
            setCurrentClassNames(Array.isArray(classes) ? classes : []);
        } catch (err) {
            console.error(err);
            setCurrentClassNames([]);
        }
    }, []);

    const fetchImages = useCallback(async (pageNum = 1) => {
        setIsLoading(true);
        const limit = mode === 'single' ? 1 : pageSize;
        const params = new URLSearchParams({ search: searchQuery, fee: feeOption, dataset: searchDataset, class: searchClass, page: pageNum, [mode === 'single' ? 'per_page' : 'limit']: limit });
        const endpoint = `${BASE_URL}/api/images/${mode}?${params.toString()}`;
        try {
            const res = await fetch(endpoint);
            if (!res.ok) throw new Error(`Fetch error ${res.status}: ${await res.text()}`);
            const result = await res.json();
            const mapped = result.images.map(item => ({ ...item, dataUrl: `data:image/jpeg;base64,${item.base64}` }));
            setImages(mapped);
            setTotalImagesCount(result.total_count);
            if (mode === 'single') {
                setCurrentIndex(0);
            }
            setHistory([]);
            setHistoryIndex(-1);
        } catch (err) {
            console.error("Fetch images error:", err);
            alert(err.message);
            setImages([]);
            setTotalImagesCount(0);
        } finally {
            setIsLoading(false);
        }
    }, [mode, pageSize, searchQuery, feeOption, searchDataset, searchClass]);
    
    useEffect(() => {
        fetchClasses(searchDataset);
        setSearchClass("");
    }, [searchDataset, fetchClasses]);
    
    useEffect(() => { fetchImages(page); }, [page, fetchImages]);

    const handleSearchSubmit = (e) => {
        if (e) e.preventDefault();
        if (page === 1) { fetchImages(1); } 
        else { setPage(1); }
    };
    
    useEffect(() => { setPage(1); }, [mode]);

    // --- AUTOSAVE LOGIC ---
    const saveAnnotations = useCallback(async (imageToSave) => {
        if (!imageToSave) return;
        setAutosaveStatus("Saving...");
        try {
            const { dataset_name, filename, boxes, original_width, original_height } = imageToSave;
            const body = {
                boxes,
                imageWidth: original_width,
                imageHeight: original_height
            };
            const res = await fetch(`${BASE_URL}/api/annotations/${dataset_name}/${filename}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            if (!res.ok) throw new Error(await res.text());
            setAutosaveStatus("Saved");
        } catch (err) {
            console.error("Autosave error:", err);
            setAutosaveStatus("Error");
        }
    }, []);

    const triggerAutosave = useCallback((imageToSave) => {
        if (autosaveTimeoutRef.current) {
            clearTimeout(autosaveTimeoutRef.current);
        }
        autosaveTimeoutRef.current = setTimeout(() => {
            saveAnnotations(imageToSave);
        }, AUTOSAVE_DELAY);
    }, [saveAnnotations]);

    // --- SINGLE MODE ANNOTATION LOGIC ---
    const saveStateForUndo = useCallback((updatedBoxes) => {
        const newHistory = history.slice(0, historyIndex + 1);
        setHistory([...newHistory, updatedBoxes]);
        setHistoryIndex(prevIndex => prevIndex + 1);
    }, [history, historyIndex]);

    const handleUndo = useCallback(() => {
        if (historyIndex > 0) {
            const prevBoxes = history[historyIndex - 1];
            setImages(prev => {
                const updated = [...prev];
                updated[currentIndex] = { ...updated[currentIndex], boxes: prevBoxes };
                triggerAutosave(updated[currentIndex]);
                return updated;
            });
            setHistoryIndex(prevIndex => prevIndex - 1);
        }
    }, [history, historyIndex, currentIndex, triggerAutosave]);

    const drawSingleCanvasContent = useCallback((ctx, img, tempBox = null) => {
        ctx.clearRect(0, 0, SINGLE_MODE_CANVAS_WIDTH, SINGLE_MODE_CANVAS_HEIGHT);
        const { original_width: originalWidth, original_height: originalHeight } = img;
        if (!originalWidth || !originalHeight) return;
        const aspectRatio = originalWidth / originalHeight;
        let drawWidth, drawHeight;
        if (aspectRatio > (SINGLE_MODE_CANVAS_WIDTH / SINGLE_MODE_CANVAS_HEIGHT)) {
            drawWidth = SINGLE_MODE_CANVAS_WIDTH;
            drawHeight = drawWidth / aspectRatio;
        } else {
            drawHeight = SINGLE_MODE_CANVAS_HEIGHT;
            drawWidth = drawHeight * aspectRatio;
        }
        const offsetX = (SINGLE_MODE_CANVAS_WIDTH - drawWidth) / 2;
        const offsetY = (SINGLE_MODE_CANVAS_HEIGHT - drawHeight) / 2;
        const scaleX = drawWidth / originalWidth;
        const scaleY = drawHeight / originalHeight;
        ctx.drawImage(imgCacheRef.current, offsetX, offsetY, drawWidth, drawHeight);
        img.boxes.forEach(box => drawBox(ctx, box, offsetX, offsetY, scaleX, scaleY, false, false, currentClassNames));
        if (tempBox) drawBox(ctx, tempBox, offsetX, offsetY, scaleX, scaleY, true, false, currentClassNames);
    }, [currentClassNames]);

    useEffect(() => {
        if (mode === 'single' && images.length > 0) {
            const img = images[currentIndex];
            const canvas = singleCanvasRef.current;
            const ctx = canvas.getContext('2d');
            if (imgCacheRef.current.src !== img.dataUrl) {
                imgCacheRef.current.src = img.dataUrl;
                imgCacheRef.current.onload = () => drawSingleCanvasContent(ctx, img);
            } else {
                drawSingleCanvasContent(ctx, img);
            }
        }
    }, [mode, images, currentIndex, drawSingleCanvasContent]);

    const getOriginalCoordsSingle = (e) => {
        const canvas = singleCanvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const img = images[currentIndex];
        const { original_width: originalWidth, original_height: originalHeight } = img;
        if (!originalWidth || !originalHeight) return null;
        const aspectRatio = originalWidth / originalHeight;
        let drawWidth, drawHeight;
        if (aspectRatio > (SINGLE_MODE_CANVAS_WIDTH / SINGLE_MODE_CANVAS_HEIGHT)) {
            drawWidth = SINGLE_MODE_CANVAS_WIDTH;
            drawHeight = drawWidth / aspectRatio;
        } else {
            drawHeight = SINGLE_MODE_CANVAS_HEIGHT;
            drawWidth = drawHeight * aspectRatio;
        }
        const offsetX = (SINGLE_MODE_CANVAS_WIDTH - drawWidth) / 2;
        const offsetY = (SINGLE_MODE_CANVAS_HEIGHT - drawHeight) / 2;
        const canvasX = (e.clientX - rect.left) * (SINGLE_MODE_CANVAS_WIDTH / rect.width);
        const canvasY = (e.clientY - rect.top) * (SINGLE_MODE_CANVAS_HEIGHT / rect.height);
        return { x: (canvasX - offsetX) * (originalWidth / drawWidth), y: (canvasY - offsetY) * (originalHeight / drawHeight) };
    };

    const handleSingleMouseDown = (e) => {
        if (e.button !== 0) return;
        setIsDrawing(true);
        setStartPt(getOriginalCoordsSingle(e));
    };

    const handleSingleMouseMove = (e) => {
        if (!isDrawing || !startPt) return;
        const canvas = singleCanvasRef.current;
        const ctx = canvas.getContext("2d");
        const currentPt = getOriginalCoordsSingle(e);
        if (!currentPt) return;
        const tempBox = { x: startPt.x, y: startPt.y, w: currentPt.x - startPt.x, h: currentPt.y - startPt.y, classId: currentClassId };
        drawSingleCanvasContent(ctx, images[currentIndex], tempBox);
    };

    const handleSingleMouseUp = (e) => {
        if (!isDrawing || !startPt) return;
        setIsDrawing(false);
        const endPt = getOriginalCoordsSingle(e);
        if (!endPt) { setStartPt(null); return; }
        let [x1, y1, x2, y2] = [startPt.x, startPt.y, endPt.x, endPt.y];
        const newBox = { classId: currentClassId, x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x1 - x2), h: Math.abs(y1 - y2) };
        if (newBox.w > 2 && newBox.h > 2) {
            const updatedBoxes = [...images[currentIndex].boxes, newBox];
            saveStateForUndo(updatedBoxes);
            setImages(prev => {
                const updated = [...prev];
                updated[currentIndex] = { ...updated[currentIndex], boxes: updatedBoxes };
                triggerAutosave(updated[currentIndex]);
                return updated;
            });
        }
        setStartPt(null);
    };

    // --- JSX RENDERING ---
    return (
        <div style={styles.container}>
            {isLoading && (
                <div style={styles.loadingOverlay}>
                    <div style={styles.spinner}></div>
                    <p style={{color: '#343A40'}}>Loading...</p>
                </div>
            )}

            <header style={styles.header}>
                <h1 style={styles.headerTitle}>🖼️ Annotation Tool</h1>
                <div style={styles.autosaveStatus}>
                    {autosaveStatus}
                </div>
            </header>

            <main style={styles.main}>
                <div style={styles.leftPanel}>
                    <div style={styles.card}>
                        <h2 style={styles.cardTitle}>Upload Dataset</h2>
                        <form onSubmit={handleUploadSubmit} style={styles.form}>
                            <input type="text" value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="Enter New Dataset Name" required style={styles.input} />
                            <input type="file" multiple onChange={(e) => setSelectedFiles(Array.from(e.target.files))} accept=".zip,image/*" style={styles.input} />
                            <button type="submit" disabled={isLoading} style={isLoading ? styles.buttonDisabled : styles.button}>Upload</button>
                        </form>
                    </div>

                    <div style={styles.card}>
                        <h2 style={styles.cardTitle}>Filter & Search</h2>
                        <form onSubmit={handleSearchSubmit} style={styles.form}>
                            <input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Search by filename" style={styles.input} />
                            <input value={searchDataset} onChange={(e) => setSearchDataset(e.target.value)} placeholder="Filter by dataset" style={styles.input} />
                            <select value={searchClass} onChange={(e) => setSearchClass(e.target.value)} style={currentClassNames.length === 0 ? { ...styles.input, ...styles.inputDisabled } : styles.input} disabled={currentClassNames.length === 0}>
                                <option value="">{currentClassNames.length > 0 ? "Filter by class..." : "Select a dataset to see classes"}</option>
                                {currentClassNames.map((name) => (<option key={name} value={name}>{name}</option>))}
                            </select>
                            <select value={feeOption} onChange={(e) => setFeeOption(e.target.value)} style={styles.input}>
                                <option value="">All</option>
                                <option value="Obj Det">Obj Det</option>
                            </select>
                            <button type="submit" disabled={isLoading} style={isLoading ? styles.buttonDisabled : styles.button}>Search</button>
                        </form>
                    </div>
                </div>

                <div style={styles.rightPanel}>
                    <div style={styles.card}>
                        <div style={styles.workspaceHeader}>
                            <h2 style={styles.cardTitle}>Workspace</h2>
                            <div style={styles.viewToggle}>
                                <button onClick={() => setMode("single")} style={mode === 'single' ? styles.buttonActive : styles.button}>Single</button>
                                <button onClick={() => setMode("multiple")} style={mode === 'multiple' ? styles.buttonActive : styles.button}>Multiple</button>
                            </div>
                        </div>
                        
                        <div style={styles.workspaceLayout}>
                            <div style={styles.workspaceContentWrapper}>
                                {mode === 'single' && (
                                    <div style={styles.annotationControls}>
                                        <button onClick={handleUndo} disabled={historyIndex <= 0} style={historyIndex <= 0 ? styles.buttonDisabled : styles.button}>Undo</button>
                                    </div>
                                )}
                                
                                <div style={styles.workspaceContent}>
                                    {images.length === 0 && !isLoading && <p>No images found. Try adjusting your filters or uploading a new dataset.</p>}
                                    
                                    {mode === 'single' && images.length > 0 && (
                                        <div style={styles.singleViewContainer}>
                                            <canvas ref={singleCanvasRef} width={SINGLE_MODE_CANVAS_WIDTH} height={SINGLE_MODE_CANVAS_HEIGHT} style={styles.singleCanvas} onMouseDown={handleSingleMouseDown} onMouseMove={handleSingleMouseMove} onMouseUp={handleSingleMouseUp}/>
                                        </div>
                                    )}

                                    {mode === 'multiple' && images.length > 0 && (
                                        <div style={styles.imageGrid}>
                                            {images.map((img, idx) => (
                                                <MultipleThumb 
                                                    key={img.id} 
                                                    image={img} 
                                                    classId={currentClassId} 
                                                    onUpdateImage={(updatedImage) => {
                                                        const newImages = [...images];
                                                        newImages[idx] = updatedImage;
                                                        setImages(newImages);
                                                        triggerAutosave(updatedImage);
                                                    }} 
                                                    currentClassNames={currentClassNames}
                                                />
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                            <ClassesPanel 
                                classNames={currentClassNames}
                                currentClassId={currentClassId}
                                onClassSelect={setCurrentClassId}
                            />
                        </div>
                        
                        <div style={styles.pagination}>
                            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1 || isLoading} style={(page <= 1 || isLoading) ? styles.buttonDisabled : styles.button}>Previous</button>
                            <span> Page {page} of {Math.ceil(totalImagesCount / (mode === 'single' ? 1 : pageSize))} (Total: {totalImagesCount}) </span>
                            <button onClick={() => setPage(p => p + 1)} disabled={(page * (mode === 'single' ? 1 : pageSize)) >= totalImagesCount || isLoading} style={((page * (mode === 'single' ? 1 : pageSize)) >= totalImagesCount || isLoading) ? styles.buttonDisabled : styles.button}>Next</button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

// --- STYLES (CSS-in-JS) ---
const styles = {
    container: { fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", backgroundColor: "#F8F9FA", minHeight: "100vh" },
    header: { backgroundColor: "#FFFFFF", padding: "1rem 2rem", borderBottom: "1px solid #DEE2E6", boxShadow: "0 2px 4px rgba(0,0,0,0.05)", display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
    headerTitle: { color: '#D32F2F', margin: 0, fontWeight: 600 },
    autosaveStatus: { color: '#6C757D', fontSize: '0.9rem', fontStyle: 'italic' },
    main: { display: "flex", padding: "1.5rem", gap: "1.5rem", alignItems: 'flex-start' },
    leftPanel: { flex: "0 0 350px", display: "flex", flexDirection: "column", gap: "1.5rem" },
    rightPanel: { flex: 1, minWidth: 0 },
    card: { backgroundColor: "#FFFFFF", borderRadius: "8px", padding: "1.5rem", boxShadow: "0 1px 3px rgba(0,0,0,0.08)" },
    cardTitle: { marginTop: 0, marginBottom: "1.5rem", color: "#343A40", borderBottom: '1px solid #E9ECEF', paddingBottom: '0.75rem' },
    form: { display: "flex", flexDirection: "column", gap: "1rem" },
    input: { padding: "0.75rem", border: "1px solid #CED4DA", borderRadius: "4px", fontSize: "1rem", backgroundColor: "#FFFFFF", transition: 'border-color 0.2s, box-shadow 0.2s' },
    inputDisabled: { backgroundColor: "#E9ECEF", color: "#6C757D", cursor: "not-allowed" },
    button: { padding: "0.75rem 1.5rem", border: "none", borderRadius: "4px", backgroundColor: "#D32F2F", color: "white", fontSize: "1rem", cursor: "pointer", transition: "background-color 0.2s", fontWeight: '500' },
    buttonActive: { padding: "0.75rem 1.5rem", border: "none", borderRadius: "4px", backgroundColor: "#B71C1C", color: "white", fontSize: "1rem", cursor: "pointer", fontWeight: '500' },
    buttonDisabled: { padding: "0.75rem 1.5rem", border: "none", borderRadius: "4px", backgroundColor: "#E9ECEF", color: "#6C757D", cursor: "not-allowed", fontWeight: '500' },
    workspaceHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: 'wrap', gap: '1rem' },
    viewToggle: { display: "flex", gap: "0.5rem" },
    workspaceLayout: { display: 'flex', gap: '1.5rem', marginTop: '1rem' },
    workspaceContentWrapper: { flex: 1, minWidth: 0 },
    workspaceContent: { minHeight: "60vh", padding: "1rem", border: "1px dashed #DEE2E6", borderRadius: "4px", backgroundColor: '#F8F9FA' },
    imageGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(600px, 1fr))", gap: "1.5rem" },
    singleViewContainer: { display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', height: '75vh' },
    singleCanvas: { maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', cursor: 'crosshair', borderRadius: '4px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)' },
    pagination: { marginTop: "1.5rem", display: "flex", justifyContent: "center", alignItems: "center", gap: "1rem", color: '#495057' },
    loadingOverlay: { position: "fixed", top: 0, left: 0, width: "100%", height: "100%", backgroundColor: "rgba(255,255,255,0.8)", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", zIndex: 1000 },
    spinner: { border: "8px solid #f3f3f3", borderTop: "8px solid #D32F2F", borderRadius: "50%", width: "60px", height: "60px", animation: "spin 1s linear infinite", marginBottom: "1rem" },
    thumb: {
        card: { display: "flex", flexDirection: "column", alignItems: "center", backgroundColor: "#FFFFFF", borderRadius: "8px", boxShadow: "0 2px 4px rgba(0,0,0,0.1)", overflow: "hidden", transition: 'transform 0.2s, box-shadow 0.2s' },
        canvasContainer: { width: "100%", aspectRatio: "1 / 1", borderBottom: "1px solid #E9ECEF" },
        canvas: { display: "block", width: "100%", height: "100%", cursor: 'crosshair' },
        info: { padding: "0.75rem", width: "100%", textAlign: "center", backgroundColor: '#F8F9FA' },
        filename: { margin: 0, fontWeight: "500", color: '#495057', wordBreak: "break-all" },
    },
    annotationControls: { display: 'flex', gap: '1rem', alignItems: 'center', paddingBottom: '1rem', flexWrap: 'wrap' },
    classesPanel: {
        container: { 
            flex: '0 0 200px', 
            borderLeft: '1px solid #DEE2E6', 
            paddingLeft: '1.5rem',
            position: 'sticky',
            top: '1.5rem',
            alignSelf: 'flex-start',
            height: 'calc(100vh - 10rem)',
            display: 'flex',
            flexDirection: 'column',
        },
        title: { marginTop: 0, marginBottom: '1rem', color: '#495057', fontSize: '1.1rem' },
        list: { 
            flex: 1,
            minHeight: 0,
            overflowY: 'auto', 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '0.5rem' 
        },
        button: { width: '100%', padding: '0.6rem 1rem', textAlign: 'left', border: '1px solid #CED4DA', backgroundColor: '#FFFFFF', borderRadius: '4px', cursor: 'pointer', transition: 'background-color 0.2s, border-color 0.2s' },
        buttonActive: { width: '100%', padding: '0.6rem 1rem', textAlign: 'left', border: '1px solid #B71C1C', backgroundColor: '#FBE9E7', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', color: '#B71C1C' },
        placeholder: { color: '#6C757D', fontSize: '0.9rem' }
    }
};

const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = `
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .button-style:hover { background-color: #B71C1C !important; }
    .input-style:focus { border-color: #D32F2F; box-shadow: 0 0 0 2px rgba(211, 47, 47, 0.25); }
    .thumb-card:hover { transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
`;
document.head.appendChild(styleSheet);

// Add class names to elements for CSS targeting
styles.button.className = 'button-style';
styles.input.className = 'input-style';
styles.thumb.card.className = 'thumb-card';


export default ImageFilter;
