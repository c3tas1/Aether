import React, { useState, useEffect, useRef } from "react";
import Select from 'react-select';
import Modal from './Modal';
import FileTree from './FileTree';
import './ImageFilter.css';

// Constants
const BASE_URL = "http://127.0.0.1:5000";
const SINGLE_LOGICAL_WIDTH = 1200;
const SINGLE_LOGICAL_HEIGHT = 1600;
const MULTIPLE_LOGICAL_WIDTH = 160;
const MULTIPLE_LOGICAL_HEIGHT = 160;
const CLASS_NAMES = [
    "person", "car", "dog", "cat", "bottle", "chair", "tv", "phone",
    "book", "cup", "laptop", "boat", "bird", "plane", "cow", "sheep"
];
const MODEL_OPTIONS = [
    { value: 'yolov8n', label: 'YOLOv8n' },
    { value: 'yolov8s', label: 'YOLOv8s' },
    { value: 'yolov8m', label: 'YOLOv8m' },
];

function ImageFilter() {
    // --- State Hooks ---
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [lastUploadedDatasetId, setLastUploadedDatasetId] = useState(null);
    
    // Modal and Dataset State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [datasetName, setDatasetName] = useState('');
    const [datasetType, setDatasetType] = useState('');
    const [datasetDescription, setDatasetDescription] = useState('');
    const [datasetClasses, setDatasetClasses] = useState(CLASS_NAMES.join(', '));
    
    // Path selection state
    const [activePathSelection, setActivePathSelection] = useState(null); // 'train', 'valid', or 'test'
    const [trainPath, setTrainPath] = useState('');
    const [validPath, setValidPath] = useState('');
    const [testPath, setTestPath] = useState('');

    // Upload/Decompression State
    const [modalView, setModalView] = useState('form'); // 'form', 'uploading', 'decompressing', 'complete'
    const [uploadProgress, setUploadProgress] = useState(0);
    const [decompressionProgress, setDecompressionProgress] = useState(0);
    const [fileTree, setFileTree] = useState(null);
    
    // Other existing state...
    const [searchQuery, setSearchQuery] = useState("");
    const [feeOption, setFeeOption] = useState("");
    const [mode, setMode] = useState("single");
    const [images, setImages] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(25);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);
    const [currentClassId, setCurrentClassId] = useState(0);
    const [selectedModel, setSelectedModel] = useState(null);
    const singleCanvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const pollingIntervalRef = useRef(null);

    // --- Handlers ---
    const handleFileChange = (e) => {
        setSelectedFiles(Array.from(e.target.files));
    };

    const handleOpenUploadModal = () => {
        if (selectedFiles.length === 0) {
            alert("Please select files before uploading.");
            return;
        }
        setModalView('form');
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
        }
        // Reset all modal-related state
        setModalView('form');
        setFileTree(null);
        setActivePathSelection(null);
        setDatasetName('');
        setDatasetType('');
        setDatasetDescription('');
        setDatasetClasses(CLASS_NAMES.join(', '));
        setTrainPath('');
        setValidPath('');
        setTestPath('');
    };
    
    const pollDecompressionStatus = (datasetId) => {
        pollingIntervalRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${BASE_URL}/api/datasets/${datasetId}/status`);
                const data = await res.json();

                if (data.status === 'decompressing') {
                    setDecompressionProgress(data.progress);
                } else if (data.status === 'complete') {
                    clearInterval(pollingIntervalRef.current);
                    const treeRes = await fetch(`${BASE_URL}/api/datasets/${datasetId}/file_tree`);
                    const treeData = await treeRes.json();
                    setFileTree(treeData);
                    setModalView('complete');
                } else if (data.status === 'failed') {
                    clearInterval(pollingIntervalRef.current);
                    alert('Decompression failed on the server.');
                    setModalView('form');
                }
            } catch (error) {
                console.error("Polling error:", error);
                clearInterval(pollingIntervalRef.current);
            }
        }, 2000);
    };

    const handleConfirmUpload = (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('datasetName', datasetName);
        formData.append('datasetType', datasetType);
        formData.append('datasetDescription', datasetDescription);
        formData.append("images", selectedFiles[0]);

        setModalView('uploading');
        setUploadProgress(0);

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener("progress", (event) => {
            if (event.lengthComputable) {
                setUploadProgress(Math.round((event.loaded / event.total) * 100));
            }
        });

        xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                const result = JSON.parse(xhr.responseText);
                setLastUploadedDatasetId(result.dataset_id);
                setModalView('decompressing');
                pollDecompressionStatus(result.dataset_id);
            } else {
                alert(`Upload failed: ${xhr.statusText}`);
                setModalView('form');
            }
        });
        xhr.addEventListener("error", () => {
            alert("An error occurred during the upload.");
            setModalView('form');
        });
        xhr.open("POST", `${BASE_URL}/api/upload`);
        xhr.send(formData);
    };
    
    const handlePreviewDataset = async () => {
        if (!lastUploadedDatasetId) {
            alert("Could not find the dataset ID.");
            return;
        }
        try {
            setSearchQuery("");
            setFeeOption("");
            setPage(1);

            const res = await fetch(`${BASE_URL}/api/datasets/${lastUploadedDatasetId}/images`);
            if (!res.ok) {
                throw new Error("Failed to fetch dataset images.");
            }
            const data = await res.json();
            const mapped = data.map(item => ({...item, dataUrl: "data:image/jpeg;base64," + item.base64}));

            setImages(mapped);
            setMode('multiple');
            handleCloseModal();

        } catch (err) {
            console.error(err);
            alert(err.message);
        }
    };

    const selectPathFor = (pathType) => {
        setActivePathSelection(pathType);
    };

    const handleFileSelect = (filePath) => {
        if (!activePathSelection) {
            alert("Please click a 'Set' button first to choose which path to assign.");
            return;
        }
        
        const fullPath = `${fileTree.name}/${filePath}`;

        if (activePathSelection === 'train') setTrainPath(fullPath);
        else if (activePathSelection === 'valid') setValidPath(fullPath);
        else if (activePathSelection === 'test') setTestPath(fullPath);
        
        setActivePathSelection(null);
    };

    const handleSaveMetadata = async () => {
        if (!lastUploadedDatasetId) {
            alert("Cannot save, dataset ID is missing.");
            return;
        }
        try {
            const res = await fetch(`${BASE_URL}/api/datasets/${lastUploadedDatasetId}/metadata`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ trainPath, validPath, testPath }),
            });
            if (!res.ok) {
                throw new Error("Failed to save metadata.");
            }
            alert("Metadata saved successfully!");
        } catch (err) {
            console.error(err);
            alert(err.message);
        }
    };

    const fetchImages = async () => {
        try {
            let endpoint = "";
            if (mode === "single") {
                endpoint = `${BASE_URL}/api/images/single?search=${searchQuery}&fee=${feeOption}&page=${page}&per_page=${pageSize}`;
            } else {
                endpoint = `${BASE_URL}/api/images/multiple?search=${searchQuery}&fee=${feeOption}&page=${page}&limit=${pageSize}`;
            }
            const res = await fetch(endpoint);
            if (!res.ok) throw new Error("Fetch error " + res.status);
            const data = await res.json();
            const mapped = (Array.isArray(data) ? data : []).map((item) => ({
                id: item.id, filename: item.filename, dataUrl: "data:image/jpeg;base64," + item.base64,
                status: item.status || "", boxes: item.boxes || [],
            }));
            setImages(mapped);
            setCurrentIndex(0);
        } catch (err) {
            console.error("Fetch images error:", err);
            alert("Error loading images. Check console.");
            setImages([]);
        }
    };

    useEffect(() => {
        if (searchQuery || feeOption) fetchImages();
    }, [mode, page, pageSize, feeOption, searchQuery]);

    useEffect(() => {
        if (mode === "single") drawSingleBoxes();
    }, [images, mode, currentIndex]);
    
    const drawSingleBoxes = () => {
        if (mode !== "single") return;
        const canvas = singleCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, SINGLE_LOGICAL_WIDTH, SINGLE_LOGICAL_HEIGHT);

        if (images.length === 0) return;
        const img = images[currentIndex];
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        img.boxes.forEach((box) => {
            ctx.strokeRect(box.x, box.y, box.w, box.h);
            const label = CLASS_NAMES[box.classId] || `id=${box.classId}`;
            ctx.fillStyle = "#00FF00";
            ctx.font = "18px 'Fira Code', monospace";
            ctx.fillText(label, box.x + 2, box.y + 18);
        });
    };

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
        if (!isDrawing) return;
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
        if (!isDrawing) return;
        setIsDrawing(false);
        const rect = e.currentTarget.getBoundingClientRect();
        const scaleX = SINGLE_LOGICAL_WIDTH / rect.width;
        const scaleY = SINGLE_LOGICAL_HEIGHT / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const w = mx - startPt.x;
        const h = my - startPt.y;
        const newBox = { classId: currentClassId, x: startPt.x, y: startPt.y, w, h };
        const copy = [...images];
        copy[currentIndex].boxes.push(newBox);
        setImages(copy);
        setStartPt(null);
    };

    const saveSingleAnnotations = async () => {
        if (images.length === 0) return;
        const img = images[currentIndex];
        try {
            const body = { boxes: img.boxes, imageWidth: SINGLE_LOGICAL_WIDTH, imageHeight: SINGLE_LOGICAL_HEIGHT };
            const res = await fetch(`${BASE_URL}/api/annotations/${img.filename}`, {
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

    const handlePrevImage = () => {
        setCurrentIndex((i) => (i === 0 ? images.length - 1 : i - 1));
    };
    
    const handleNextImage = () => {
        setCurrentIndex((i) => (i === images.length - 1 ? 0 : i + 1));
    };

    const handleDiscard = async (img) => {
        try {
            const res = await fetch(`${BASE_URL}/api/images/${img.id}/discard`, { method: "PUT" });
            if (!res.ok) throw new Error("Discard error " + res.status);
            const result = await res.json();
            alert("Discarded image_id: " + result.image_id);
            setImages((prev) => prev.map((x) => (x.id === img.id ? { ...x, status: "discarded" } : x)));
        } catch (err) {
            console.error("Discard error:", err);
            alert("Discard failed. See console.");
        }
    };

    const handleSearchSubmit = (e) => {
        e.preventDefault();
        setPage(1);
        fetchImages();
    };

    const renderModalContent = () => {
        switch (modalView) {
            case 'uploading':
            case 'decompressing':
                return (
                    <div className="progress-bar-container">
                        <div className="progress-bar-label">
                            {modalView === 'uploading' ? `Uploading... ${uploadProgress}%` : `Decompressing... ${decompressionProgress}%`}
                        </div>
                        <div className="progress-bar">
                            <div className="progress-bar-fill" style={{ width: `${modalView === 'uploading' ? uploadProgress : decompressionProgress}%` }}></div>
                        </div>
                    </div>
                );
            case 'complete':
                return (
                    <div className="dataset-info-panel">
                        <pre className="modal-header-art">{`+------------------------------------+\n| [Configure Dataset Paths]          |\n+------------------------------------+`}</pre>
                        <div className="modal-form-columns">
                            <div className="modal-form-column">
                                <div className="file-tree-instructions">
                                    Click a "Set" button, then click a file below.
                                </div>
                                <div className="file-tree-container">
                                    <FileTree 
                                        node={fileTree} 
                                        onFileClick={handleFileSelect}
                                        activePath={ (activePathSelection === 'train' && trainPath) || (activePathSelection === 'valid' && validPath) || (activePathSelection === 'test' && testPath) || '' }
                                        isRoot={true}
                                    />
                                </div>
                            </div>
                            <div className="modal-form-column path-selection-panel">
                                <div className={`path-input-group ${activePathSelection === 'train' ? 'active-selection' : ''}`}>
                                    <label>Train Data Path</label>
                                    <div className="path-input-row">
                                        <input className="terminal-input" type="text" value={trainPath} onChange={e => setTrainPath(e.target.value)} />
                                        <button type="button" onClick={() => selectPathFor('train')} className={`terminal-button small ${activePathSelection === 'train' ? 'active' : ''}`}>Set</button>
                                    </div>
                                </div>
                                <div className={`path-input-group ${activePathSelection === 'valid' ? 'active-selection' : ''}`}>
                                    <label>Valid Data Path</label>
                                    <div className="path-input-row">
                                        <input className="terminal-input" type="text" value={validPath} onChange={e => setValidPath(e.target.value)} />
                                        <button type="button" onClick={() => selectPathFor('valid')} className={`terminal-button small ${activePathSelection === 'valid' ? 'active' : ''}`}>Set</button>
                                    </div>
                                </div>
                                <div className={`path-input-group ${activePathSelection === 'test' ? 'active-selection' : ''}`}>
                                    <label>Test Data Path</label>
                                    <div className="path-input-row">
                                        <input className="terminal-input" type="text" value={testPath} onChange={e => setTestPath(e.target.value)} />
                                        <button type="button" onClick={() => selectPathFor('test')} className={`terminal-button small ${activePathSelection === 'test' ? 'active' : ''}`}>Set</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="info-panel-actions">
                            <button onClick={handleSaveMetadata} className="terminal-button primary">Save Metadata</button>
                            <button onClick={handlePreviewDataset} className="terminal-button">Preview Dataset</button>
                            <button onClick={handleCloseModal} className="terminal-button">Done</button>
                        </div>
                    </div>
                );
            case 'form':
            default:
                return (
                    <form onSubmit={handleConfirmUpload} className="modal-form">
                        <pre className="modal-header-art">{`+------------------------------------+\n| [Initialize New Dataset Sequence]  |\n+------------------------------------+`}</pre>
                        <div className="modal-form-columns">
                            <div className="modal-form-column">
                                <div className="form-group"><label htmlFor="datasetName">Dataset Name</label><input id="datasetName" className="terminal-input" type="text" value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="e.g., Road_Objects_Q3" required /></div>
                                <div className="form-group"><label htmlFor="datasetType">Dataset Type</label><input id="datasetType" className="terminal-input" type="text" value={datasetType} onChange={(e) => setDatasetType(e.target.value)} placeholder="e.g., Object Detection" /></div>
                                <div className="form-group"><label htmlFor="datasetDescription">Description</label><textarea id="datasetDescription" className="terminal-textarea" value={datasetDescription} onChange={(e) => setDatasetDescription(e.target.value)} rows="5" placeholder="A short description..."></textarea></div>
                            </div>
                            <div className="modal-form-column">
                                <div className="form-group"><label htmlFor="datasetClasses">Classes</label><textarea id="datasetClasses" className="terminal-textarea" value={datasetClasses} onChange={(e) => setDatasetClasses(e.target.value)} rows="5"></textarea></div>
                                <div className="form-group"><label htmlFor="trainPath">Train Data Path</label><input id="trainPath" className="terminal-input" type="text" value={trainPath} onChange={(e) => setTrainPath(e.target.value)} placeholder="/path/to/train" /></div>
                                <div className="form-group"><label htmlFor="validPath">Valid Data Path</label><input id="validPath" className="terminal-input" type="text" value={validPath} onChange={(e) => setValidPath(e.target.value)} placeholder="/path/to/valid" /></div>
                                <div className="form-group"><label htmlFor="testPath">Test Data Path</label><input id="testPath" className="terminal-input" type="text" value={testPath} onChange={(e) => setTestPath(e.target.value)} placeholder="/path/to/test" /></div>
                            </div>
                        </div>
                        <button type="submit" className="terminal-button primary full-width">Confirm & Upload</button>
                    </form>
                );
        }
    };

    return (
        <div className="image-filter-container">
            <Modal isOpen={isModalOpen} onClose={handleCloseModal}>{renderModalContent()}</Modal>
            <h1 className="page-title">{'// ANNOTATION_INTERFACE'}</h1>
            <div className="controls-grid">
                <fieldset className="control-group">
                    <legend>DATA INPUT</legend>
                    <div className="data-input-form">
                        <label htmlFor="file-upload" className="terminal-button">{selectedFiles.length > 0 ? `${selectedFiles.length} files selected` : 'SELECT FILES'}</label>
                        <input id="file-upload" type="file" multiple ref={fileInputRef} onChange={handleFileChange} accept=".zip,image/*" />
                        <button onClick={handleOpenUploadModal} className="terminal-button primary" disabled={selectedFiles.length === 0}>Upload</button>
                    </div>
                </fieldset>
                <fieldset className="control-group filter-search-panel">
                    <legend>FILTER & SEARCH</legend>
                    <form onSubmit={handleSearchSubmit}>
                        <input className="terminal-input" placeholder="Search datasets or files..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
                        <select className="terminal-select" value={feeOption} onChange={(e) => setFeeOption(e.target.value)}>
                            <option value="">FEE: ALL</option>
                            <option value="Obj Det">FEE: Obj Det</option>
                        </select>
                        <button type="submit" className="terminal-button primary">Search</button>
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
                {mode === "single" && images.length > 0 && (
                    <div className="single-mode-container">
                        <div className="single-image-wrapper">
                            <img src={images[currentIndex].dataUrl} alt="Annotation target" className="main-image" />
                            <canvas ref={singleCanvasRef} width={SINGLE_LOGICAL_WIDTH} height={SINGLE_LOGICAL_HEIGHT} className="annotation-canvas" onMouseDown={handleSingleMouseDown} onMouseMove={handleSingleMouseMove} onMouseUp={handleSingleMouseUp} />
                        </div>
                        <div className="action-bar">
                            <button className="terminal-button" onClick={saveSingleAnnotations}>Save Annotations</button>
                            <button className="terminal-button discard" onClick={() => handleDiscard(images[currentIndex])}>{images[currentIndex].status === "discarded" ? "Discarded" : "Discard"}</button>
                            <button className="terminal-button" onClick={handlePrevImage}>Prev</button>
                            <button className="terminal-button" onClick={handleNextImage}>Next</button>
                        </div>
                    </div>
                )}
                {mode === "single" && images.length === 0 && (<p className="status-text">No images found for single mode.</p>)}
                {mode === "multiple" && (
                    <div className="multiple-mode-container">
                        {images.length > 0 ? (
                            <div className="multiple-mode-grid">
                                {images.map((img, idx) => (
                                    <MultipleThumb key={img.id} image={img} classId={currentClassId} onDiscard={() => handleDiscard(img)} onUpdateImage={(updated) => {
                                        setImages((prev) => {
                                            const copy = [...prev];
                                            copy[idx] = updated;
                                            return copy;
                                        });
                                    }} />
                                ))}
                            </div>
                        ) : (<p className="status-text">No images found for multiple mode.</p>)}
                        <div className="action-bar">
                            <label>Images per page:</label>
                            <select className="terminal-select" value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}>
                                <option value={25}>25</option>
                                <option value={50}>50</option>
                                <option value={100}>100</option>
                            </select>
                        </div>
                        <div className="pagination-controls">
                            <button className="terminal-button" onClick={() => setPage((p) => (p > 1 ? p - 1 : p))} disabled={page <= 1}>Previous</button>
                            <span>Page {page}</span>
                            <button className="terminal-button" onClick={() => setPage((p) => p + 1)}>Next</button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function MultipleThumb({ image, classId, onDiscard, onUpdateImage }) {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPt, setStartPt] = useState(null);

    useEffect(() => { drawBoxes(); }, [image.boxes]);

    const drawBoxes = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, MULTIPLE_LOGICAL_WIDTH, MULTIPLE_LOGICAL_HEIGHT);
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 1;
        image.boxes.forEach((b) => {
            ctx.strokeRect(b.x, b.y, b.w, b.h);
            const label = CLASS_NAMES[b.classId] || `?`;
            ctx.fillStyle = "#00FF00";
            ctx.font = "12px 'Fira Code', monospace";
            ctx.fillText(label, b.x + 2, b.y + 12);
        });
    };

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
        const updated = { ...image, boxes: [...image.boxes, newBox] };
        onUpdateImage(updated);
        setStartPt(null);
    };

    const handleSave = async () => {
        try {
            const body = { boxes: image.boxes, imageWidth: MULTIPLE_LOGICAL_WIDTH, imageHeight: MULTIPLE_LOGICAL_HEIGHT };
            const res = await fetch(`${BASE_URL}/api/annotations/${image.filename}`, {
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
                <img src={image.dataUrl} alt={image.filename} className="main-image" />
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