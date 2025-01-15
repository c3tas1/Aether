import React, { useState } from "react";

const Upload = () => {
    const [selectedImages, setSelectedImages] = useState([]);
    const [imageUrls, setImageUrls] = useState([]);
    
    const handleImageChange = (event) => {
        const files = Array.from(event.target.files);
        setSelectedImages(files);
    
        const urls = [];
        files.forEach((file) => {
        const reader = new FileReader();
        reader.onload = () => { // Use onload instead of onloadend
            urls.push(reader.result);
            if (urls.length === files.length) { // Check if all URLs are generated
            setImageUrls(urls); 
            }
        };
        reader.readAsDataURL(file);
        });
    };
    
    const handleUpload = async () => {
        if (!selectedImages.length) {
        alert('Please select at least one image.');
        return;
        }
    
        try {
        // Replace with your actual upload logic for multiple images
        console.log('Uploading images:', selectedImages);
        // ... (your upload logic using FormData)
    
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
        <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh', 
        backgroundColor: '#f4f4f4' // Example background color
        }}>
        <div style={{ 
            border: '1px solid #ccc', 
            padding: '30px', 
            borderRadius: '5px', 
            backgroundColor: '#fff',
            textAlign: 'center' 
        }}>
            <input type="file" accept="image/*" multiple onChange={handleImageChange} />
    
            <div style={{ 
            display: 'flex', 
            flexWrap: 'wrap', 
            marginTop: '20px', 
            justifyContent: 'center' 
            }}>
            {imageUrls.map((imageUrl, index) => (
                <img 
                key={index} 
                src={imageUrl} 
                alt={`Uploaded preview ${index + 1}`} 
                style={{ maxWidth: '200px', margin: '10px' }} 
                />
            ))}
            </div>
    
            <button 
            onClick={handleUpload} 
            disabled={!selectedImages.length} 
            style={{ 
                marginTop: '20px', 
                padding: '10px 20px', 
                cursor: 'pointer',
                backgroundColor: '#007bff', // Example button color
                color: '#fff',
                border: 'none',
                borderRadius: '5px'
            }}
            >
            Upload
            </button>
        </div>
        </div>
    );
}
export default Upload;