#!/usr/bin/env python3
"""
Dataset Download Script for dyGRASS Dynamic Graph Sparsification

This script downloads the preprocessed datasets required for running
dynamic graph sparsification experiments. The datasets include:
- Dense and sparse graph representations (MTX format)
- Streaming edge update batches (incremental/decremental)
- Multiple graph types: network topology, finite element meshes, etc.

Usage:
    python datasetdownload.py

Requirements:
    pip install gdown
"""

import gdown
import tarfile
import pathlib
import sys
import os

# Configuration
FILE_ID = "1gcX2WmO_XaQRCUbCJz40uugR77Jp7Y1Y"  # Your dataset file ID
ARCHIVE = "dataset.tar.xz"                      # Using .tar.xz for better compression
TARGET = pathlib.Path(".")                  # Extract to current directory

def download_and_extract():
    """Download and extract the dyGRASS datasets"""
    
    print("🔄 Downloading dyGRASS datasets...")
    print(f"   File ID: {FILE_ID}")
    print(f"   Archive: {ARCHIVE}")
    
    # Download the dataset archive
    try:
        gdown.download(id=FILE_ID, output=ARCHIVE, quiet=False)
    except Exception as e:
        sys.exit(f"❌ Download failed: {e}")
    
    # Verify the archive
    if not os.path.exists(ARCHIVE):
        sys.exit("❌ Download failed: archive file not found")
    
    # Determine archive format and extract
    print(f"🔄 Extracting {ARCHIVE}...")
    
    try:
        if ARCHIVE.endswith('.tar.xz'):
            with tarfile.open(ARCHIVE, "r:xz") as tar:
                tar.extractall(TARGET)
        elif ARCHIVE.endswith('.tar.gz'):
            with tarfile.open(ARCHIVE, "r:gz") as tar:
                tar.extractall(TARGET)
        else:
            sys.exit("❌ Unsupported archive format")
            
    except tarfile.ReadError:
        sys.exit("❌ Extraction failed: not a valid archive (check sharing permissions or file ID)")
    except Exception as e:
        sys.exit(f"❌ Extraction failed: {e}")
    
    # Clean up archive file
    os.remove(ARCHIVE)
    print(f"🗑️  Removed archive file: {ARCHIVE}")
    
    # Verify extraction
    dataset_path = TARGET / "dataset"
    if dataset_path.exists():
        # Count available datasets
        datasets = [d for d in dataset_path.iterdir() if d.is_dir()]
        print(f"✅ Datasets ready: {len(datasets)} graphs available")
        
        # List available datasets
        print("\n📊 Available datasets:")
        for dataset in sorted(datasets):
            # Check if required files exist
            dense_file = dataset / "new_adj_dense.mtx"
            sparse_file = dataset / "new_adj_sparse.mtx"
            stream_dir = dataset / "stream_edges"
            
            if dense_file.exists() and sparse_file.exists() and stream_dir.exists():
                batch_count = len(list(stream_dir.glob("*.mtx")))
                print(f"   - {dataset.name:12} ({batch_count} batches)")
            else:
                print(f"   - {dataset.name:12} (⚠️  incomplete)")
    else:
        print("⚠️  Warning: dataset directory not found after extraction")

def main():
    """Main function"""
    print("=" * 60)
    print("dyGRASS Dataset Download Script")
    print("=" * 60)
    
    # Check if datasets already exist
    if (TARGET / "dataset").exists():
        response = input("📁 Dataset directory already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return
    
    # Check dependencies
    try:
        import gdown
    except ImportError:
        print("❌ Missing dependency: gdown")
        print("   Install with: pip install gdown")
        return
    
    # Download and extract
    download_and_extract()
    
    print("\n🎉 Dataset setup complete!")
    print("\n📖 For more information, see the README or documentation")

if __name__ == "__main__":
    main()