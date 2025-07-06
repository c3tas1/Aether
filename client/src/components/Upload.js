import React, { useState, useRef, useCallback } from "react";
import Modal from './Modal';
import FileTree from './FileTree';
import './Upload.css';

// Constants
const BASE_URL = "http://127.0.0.1:5000";
const CLASS_NAMES_DEFAULT = [
    "person", "car", "dog", "cat", "bottle", "chair", "tv", "phone",
    "book", "cup", "laptop", "boat", "bird", "plane", "cow", "sheep"
].join(', ');
const PYTHON_SCRIPT_TEMPLATE = `# This script generates 'manifest.json' by pairing images with their annotations.
dataset_root = '/data'

import os
import json

print("Starting indexing process with robust annotation matching...")

# --- Step 1: Find all files and categorize them ---
all_files = []
for root, dirs, files in os.walk(dataset_root):
    for name in files:
        relative_path = os.path.relpath(os.path.join(root, name), dataset_root)
        all_files.append(relative_path.replace("\\\\", "/"))

image_extensions = {'.jpg', '.jpeg', '.png'}
annotation_extensions = {'.txt', '.xml'}

# Create a map of annotations using their core filename as the key.
# e.g., 'car_01' -> 'labels/train/car_01.txt'
annotations_map = {}
for file_path in all_files:
    base_name, extension = os.path.splitext(os.path.basename(file_path))
    if extension.lower() in annotation_extensions:
        annotations_map[base_name] = file_path

# --- Step 2: Iterate through images and find their matching annotation by basename ---
manifest_data = []
for file_path in all_files:
    base_name, extension = os.path.splitext(os.path.basename(file_path))
    if extension.lower() in image_extensions:
        annotation_path = annotations_map.get(base_name, None)
        manifest_data.append({
            "image": file_path,
            "annotation": annotation_path
        })

# --- Step 3: Write the final manifest file ---
manifest_path = os.path.join(dataset_root, 'manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest_data, f, indent=2)

print(f"Successfully created manifest for {len(manifest_data)} images.")
print("Indexing complete.")
`;

// Helper component for form steps
const FormStep = ({ currentStep, stepNumber, title, children }) => {
    if (currentStep !== stepNumber) return null;
    return (
        <div className="form-step">
            <h3 className="step-title">{title}</h3>
            {children}
        </div>
    );
};

function Upload() {
    // --- State Hooks ---
    const [currentStep, setCurrentStep] = useState(1);
    const [selectedFile, setSelectedFile] = useState(null);
    const [thumbnailFile, setThumbnailFile] = useState(null);
    const [thumbnailPreview, setThumbnailPreview] = useState(null);
    const [lastUploadedStoragePath, setLastUploadedStoragePath] = useState(null);
    
    // Form State
    const [datasetName, setDatasetName] = useState('');
    const [datasetType, setDatasetType] = useState('');
    const [datasetDescription, setDatasetDescription] = useState('');
    const [classNames, setClassNames] = useState(CLASS_NAMES_DEFAULT);

    // Upload & Post-Upload State
    const [uploadProgress, setUploadProgress] = useState(0);
    const [decompressionProgress, setDecompressionProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [fileTree, setFileTree] = useState(null);
    const [indexingScript, setIndexingScript] = useState(PYTHON_SCRIPT_TEMPLATE);

    // Preview Modal State
    const [isPreviewModalOpen, setIsPreviewModalOpen] = useState(false);
    const [previewContent, setPreviewContent] = useState({ type: null, content: '', name: '' });

    const pollingIntervalRef = useRef(null);

    // --- Handlers ---
    const handleFileSelect = (files) => {
        if (files && files[0]) {
            const file = files[0];
            if (file.type.startsWith('image/') || file.name.endsWith('.zip')) {
                setSelectedFile(file);
                setUploadStatus('');
            } else {
                alert("Please select a valid file type (ZIP or an image).");
            }
        }
    };

    const handleDragOver = (e) => e.preventDefault();
    const handleDrop = (e) => {
        e.preventDefault();
        handleFileSelect(e.dataTransfer.files);
    };

    const handleThumbnailChange = (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setThumbnailFile(file);
            setThumbnailPreview(URL.createObjectURL(file));
        } else if (file) {
            alert("Thumbnail must be an image file.");
        }
    };

    const pollDecompressionStatus = useCallback(async (storagePath) => {
        pollingIntervalRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${BASE_URL}/api/datasets/${storagePath}/status`);
                const data = await res.json();
                if (data.status === 'decompressing') {
                    setUploadStatus(`Decompressing... ${data.progress}%`);
                    setDecompressionProgress(data.progress);
                } else {
                    clearInterval(pollingIntervalRef.current);
                    if (data.status === 'complete') {
                        setUploadStatus("Processing complete! Fetching file structure...");
                        const treeRes = await fetch(`${BASE_URL}/api/datasets/${storagePath}/file_tree`);
                        const treeData = await treeRes.json();
                        setFileTree(treeData);
                        setIsProcessing(false);
                        setCurrentStep(4); // Go to Step 4
                    } else {
                        throw new Error('Decompression failed on the server.');
                    }
                }
            } catch (error) {
                setUploadStatus(`Error: ${error.message}`);
                setIsProcessing(false);
                clearInterval(pollingIntervalRef.current);
            }
        }, 2000);
    }, []);

    const handleUpload = useCallback(() => {
        if (!selectedFile) return;
        setIsProcessing(true);
        setUploadProgress(0);
        setDecompressionProgress(0);
        setCurrentStep(3);

        const formData = new FormData();
        formData.append('datasetName', datasetName);
        formData.append('datasetType', datasetType);
        formData.append('datasetDescription', datasetDescription);
        formData.append('classNames', classNames);
        formData.append("images", selectedFile);
        if (thumbnailFile) formData.append("thumbnail", thumbnailFile);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${BASE_URL}/api/upload`);

        xhr.upload.addEventListener("progress", (event) => {
            if (event.lengthComputable) {
                const progress = Math.round((event.loaded / event.total) * 100);
                setUploadProgress(progress);
                setUploadStatus(`Uploading... ${progress}%`);
            }
        });

        xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                const result = JSON.parse(xhr.responseText);
                setLastUploadedStoragePath(result.storage_path);
                setUploadStatus("Upload complete. Awaiting decompression...");
                pollDecompressionStatus(result.storage_path);
            } else {
                setUploadStatus(`Upload failed: ${xhr.statusText}`);
                setIsProcessing(false);
            }
        });

        xhr.addEventListener("error", () => {
            setUploadStatus("An error occurred during the upload.");
            setIsProcessing(false);
        });

        xhr.send(formData);
    }, [selectedFile, datasetName, datasetType, datasetDescription, classNames, thumbnailFile, pollDecompressionStatus]);
    
    const handleNodeClick = useCallback(async (node) => {
        if (node.type !== 'file') return;

        const fileExt = node.name.split('.').pop().toLowerCase();
        const supportedExts = ['png', 'jpg', 'jpeg', 'json', 'xml', 'txt'];

        if (!supportedExts.includes(fileExt)) return;

        try {
            const res = await fetch(`${BASE_URL}/api/datasets/${lastUploadedStoragePath}/preview?path=${node.path}`);
            if (!res.ok) throw new Error("Could not fetch file content.");
            const data = await res.json();
            setPreviewContent(data);
            setIsPreviewModalOpen(true);
        } catch (err) {
            console.error(err);
            alert(err.message);
        }
    }, [lastUploadedStoragePath]);

    const pollScriptingStatus = useCallback((storagePath) => {
        pollingIntervalRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${BASE_URL}/api/datasets/${storagePath}/status`);
                if (!res.ok) throw new Error('Status check failed.');
                const data = await res.json();

                if (data.status === 'indexed') {
                    clearInterval(pollingIntervalRef.current);
                    setUploadStatus('Indexing complete! Redirecting...');
                    window.location.href = `/annotate?dataset=${storagePath}`;
                } else if (data.status === 'script_failed') {
                    clearInterval(pollingIntervalRef.current);
                    setUploadStatus('Script execution failed. Check backend logs for details.');
                    setIsProcessing(false);
                }
                // While status is 'processing_script', the loop continues
            } catch (error) {
                clearInterval(pollingIntervalRef.current);
                setUploadStatus(`An error occurred: ${error.message}`);
                setIsProcessing(false);
            }
        }, 3000);
    }, []);

    const handleExecuteScript = async () => {
        setIsProcessing(true);
        setUploadStatus('Executing indexing script...');
        try {
            const res = await fetch(`${BASE_URL}/api/datasets/${lastUploadedStoragePath}/execute-script`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: indexingScript }),
            });

            if (res.status !== 202) throw new Error("Failed to start script execution.");
            
            // On success (202 Accepted), start polling for completion
            pollScriptingStatus(lastUploadedStoragePath);

        } catch (error) {
            alert(`An error occurred: ${error.message}`);
            setUploadStatus(`Error: ${error.message}`);
            setIsProcessing(false);
        }
    };

    const resetForm = () => {
        setCurrentStep(1);
        setSelectedFile(null);
        setThumbnailFile(null);
        setThumbnailPreview(null);
        setLastUploadedStoragePath(null);
        setDatasetName('');
        setDatasetType('');
        setDatasetDescription('');
        setClassNames(CLASS_NAMES_DEFAULT);
        setUploadProgress(0);
        setDecompressionProgress(0);
        setUploadStatus('');
        setIsProcessing(false);
        setFileTree(null);
        setIndexingScript(PYTHON_SCRIPT_TEMPLATE);
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
        }
    };

    return (
        <div className="upload-container">
            <Modal isOpen={isPreviewModalOpen} onClose={() => setIsPreviewModalOpen(false)}>
                <div className="preview-panel">
                    <h3 className="preview-title">{previewContent.name}</h3>
                    <div className="preview-content">
                        {previewContent.type === 'image' && <img src={`data:image/jpeg;base64,${previewContent.content}`} alt="File preview" />}
                        {previewContent.type === 'text' && <pre>{previewContent.content}</pre>}
                    </div>
                </div>
            </Modal>

            <div className="upload-panel">
                <h1 className="panel-title">{'// DATASET_INITIALIZATION'}</h1>
                <p className="panel-subtitle">Follow the steps below to upload and configure a new dataset.</p>

                <div className="stepper">
                    {[1, 2, 3, 4].map((stepNumber, index) => (
                        <React.Fragment key={stepNumber}>
                            <div className={`step ${currentStep >= stepNumber ? 'active' : ''}`}>
                                <div className="step-icon">{stepNumber}</div>
                                <div className="step-label">{['Select File', 'Configure', 'Upload', 'Index'][index]}</div>
                            </div>
                            {index < 3 && <div className={`step-connector ${currentStep > stepNumber ? 'active' : ''}`}></div>}
                        </React.Fragment>
                    ))}
                </div>

                <div className="form-content">
                    {/* --- Step 1: File Selection --- */}
                    <FormStep currentStep={currentStep} stepNumber={1} title="Step 1: Select Your Dataset File">
                        <div className="dropzone" onDragOver={handleDragOver} onDrop={handleDrop}>
                            <input id="file-input" type="file" accept=".zip,image/*" onChange={(e) => handleFileSelect(e.target.files)} style={{ display: 'none' }} />
                            <label htmlFor="file-input" className="dropzone-label">
                                {selectedFile ? (
                                    <><span className="file-icon">📁</span><p><strong>{selectedFile.name}</strong></p><span className="file-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span><span className="change-file-btn">Click or Drag to Change File</span></>
                                ) : (
                                    <><span className="file-icon">☁️</span><p><strong>Drag & Drop a ZIP or Image File</strong></p><p className="dropzone-or">or</p><span className="browse-btn">Browse Local Files</span></>
                                )}
                            </label>
                        </div>
                        <button className="next-btn" disabled={!selectedFile} onClick={() => setCurrentStep(2)}>Next: Configure Dataset</button>
                    </FormStep>

                    {/* --- Step 2: Configuration --- */}
                    <FormStep currentStep={currentStep} stepNumber={2} title="Step 2: Provide Dataset Details">
                        <div className="form-grid">
                            <div className="form-group"><label htmlFor="datasetName">Dataset Name</label><input id="datasetName" type="text" value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="e.g., Road Objects Q3" required /></div>
                            <div className="form-group"><label htmlFor="datasetType">Dataset Type</label><input id="datasetType" type="text" value={datasetType} onChange={(e) => setDatasetType(e.target.value)} placeholder="e.g., Object Detection" /></div>
                        </div>
                        <div className="form-group"><label htmlFor="datasetDescription">Description</label><textarea id="datasetDescription" value={datasetDescription} onChange={(e) => setDatasetDescription(e.target.value)} rows="3" placeholder="A short summary of the dataset's contents..."></textarea></div>
                        <div className="form-group"><label htmlFor="classNames">Class Names (comma-separated)</label><textarea id="classNames" value={classNames} onChange={(e) => setClassNames(e.target.value)} rows="3"></textarea></div>
                        <div className="form-group thumbnail-upload">
                            <label>Optional Thumbnail</label>
                            <div className="thumbnail-control"><label htmlFor="thumbnail-input" className="browse-btn small">{thumbnailFile ? 'Change Thumbnail' : 'Select Thumbnail'}</label><input id="thumbnail-input" type="file" accept="image/*" onChange={handleThumbnailChange} style={{ display: 'none' }} />{thumbnailPreview && <img src={thumbnailPreview} alt="Preview" className="thumbnail-preview" />}</div>
                        </div>
                        <div className="button-group"><button className="back-btn" onClick={() => setCurrentStep(1)}>Back</button><button className="upload-btn" onClick={handleUpload} disabled={!datasetName || isProcessing}>Confirm & Upload</button></div>
                    </FormStep>

                    {/* --- Step 3: Upload Progress --- */}
                    <FormStep currentStep={currentStep} stepNumber={3} title="Step 3: Uploading & Processing">
                        <div className="progress-container">
                            <p className="status-text">{uploadStatus}</p>
                            <div className="progress-bar-wrapper"><div className="progress-bar-label">Upload</div><div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }}></div></div></div>
                            <div className="progress-bar-wrapper"><div className="progress-bar-label">Decompression</div><div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${decompressionProgress}%` }}></div></div></div>
                        </div>
                    </FormStep>

                    {/* --- Step 4: Indexing --- */}
                    <FormStep currentStep={currentStep} stepNumber={4} title="Step 4: Review and Index Dataset">
                        {isProcessing ? (
                            <div className="progress-container">
                                <p className="status-text">{uploadStatus}</p>
                                <div className="spinner"></div>
                            </div>
                        ) : (
                            <>
                                <div className="indexing-layout">
                                    <div className="file-tree-panel">
                                        <h4>Dataset Structure (Click to preview)</h4>
                                        <div className="file-tree-container">
                                            {fileTree ? <FileTree node={fileTree} onNodeClick={handleNodeClick} /> : <p>Loading file tree...</p>}
                                        </div>
                                    </div>
                                    <div className="scripting-panel">
                                        <h4>Indexing Script (Python)</h4>
                                        <textarea className="code-editor" value={indexingScript} onChange={(e) => setIndexingScript(e.target.value)} spellCheck="false" />
                                    </div>
                                </div>
                                <div className="button-group">
                                    <button className="back-btn" onClick={resetForm} disabled={isProcessing}>Start Over</button>
                                    <button className="upload-btn" onClick={handleExecuteScript} disabled={isProcessing}>
                                        Finish & Go to Annotate
                                    </button>
                                </div>
                            </>
                        )}
                    </FormStep>
                </div>
            </div>
        </div>
    );
}

export default Upload;