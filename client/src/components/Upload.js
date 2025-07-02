import React, { useState } from "react";
import './Upload.css'; // Import the new CSS file

const Upload = () => {
    const [selectedImages, setSelectedImages] = useState([]);
    const [imageUrls, setImageUrls] = useState([]);
    const [isHovering, setIsHovering] = useState(false);

    const handleImageChange = (event) => {
        const files = Array.from(event.target.files);
        setSelectedImages(files);

        const urls = files.map(file => URL.createObjectURL(file));
        setImageUrls(urls);
    };

    const handleUpload = async () => {
        if (!selectedImages.length) {
            alert('Please select at least one image.');
            return;
        }

        try {
            // This is where you would put your actual upload logic
            console.log('Uploading images:', selectedImages);

            // Clear the selected images after successful upload
            setSelectedImages([]);
            setImageUrls([]);
            alert('Images uploaded successfully!');
        } catch (error) {
            console.error('Error uploading images:', error);
            alert('Error uploading images.');
        }
    };

    return (
        <div className="upload-container">
            <div className="upload-card">
                <h2 className="upload-title">{'// Initialize Upload Sequence'}</h2>
                
                {/* Custom-styled file input */}
                <label 
                    htmlFor="file-input" 
                    className="upload-dropzone"
                    onDragEnter={() => setIsHovering(true)}
                    onDragLeave={() => setIsHovering(false)}
                    onDrop={() => setIsHovering(false)}
                >
                    <input 
                        id="file-input" 
                        type="file" 
                        accept="image/*" 
                        multiple 
                        onChange={handleImageChange} 
                    />
                    <p>{isHovering ? 'Release to select files' : 'Drag & drop files here, or click to select'}</p>
                    <span className="browse-btn">Browse Files</span>
                </label>
                
                {/* Image preview section */}
                {imageUrls.length > 0 && (
                    <div className="image-preview-container">
                        {imageUrls.map((imageUrl, index) => (
                            <img
                                key={index}
                                src={imageUrl}
                                alt={`Preview ${index + 1}`}
                                className="preview-image"
                            />
                        ))}
                    </div>
                )}
                
                {/* Upload button */}
                <button
                    onClick={handleUpload}
                    disabled={!selectedImages.length}
                    className="upload-button"
                >
                    Execute Upload
                </button>
            </div>
        </div>
    );
}

export default Upload;