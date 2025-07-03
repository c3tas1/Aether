import React, { useState } from 'react';
import './FileTree.css';

const FileIcon = () => ( <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M13,9V3.5L18.5,9H13Z" /></svg> );
const FolderIcon = ({ isOpen }) => ( <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">{isOpen ? <path d="M19,20H5A2,2 0 0,1 3,18V6A2,2 0 0,1 5,4H9L11,6H19A2,2 0 0,1 21,8H21V18A2,2 0 0,1 19,20M19,8H5V18H19V8Z" /> : <path d="M10,4H4C2.89,4 2,4.89 2,6V18A2,2 0 0,0 4,20H20A2,2 0 0,0 22,18V8C22,6.89 21.1,6 20,6H12L10,4Z" />}</svg> );

const FileTree = ({ node, onNodeClick, isRoot = false }) => {
    const [isOpen, setIsOpen] = useState(isRoot);

    if (!node) return null;

    const isFolder = node.type === 'folder';

    const handleNodeClick = (e) => {
        e.stopPropagation(); 
        if (onNodeClick) {
            onNodeClick(node);
        }
        if (isFolder) {
            setIsOpen(!isOpen);
        }
    };

    return (
        <div className="file-tree-node">
            <div className='node-item' onClick={handleNodeClick}>
                <span className="node-icon">
                    {isFolder ? <FolderIcon isOpen={isOpen} /> : <FileIcon />}
                </span>
                <span className="node-name">{node.name}</span>
            </div>
            {isFolder && isOpen && (
                <div className="node-children">
                    {node.children && node.children.map((childNode, index) => (
                        <FileTree 
                            key={index} 
                            node={childNode} 
                            onNodeClick={onNodeClick}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

export default FileTree;