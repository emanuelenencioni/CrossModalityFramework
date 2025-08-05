#!/usr/bin/env python3
"""
Example usage script for Cityscapes bounding box visualization.
This shows how to run the visualization scripts with your specific dataset.
"""

import os
import subprocess
import sys

def run_simple_visualization():
    """Run the simple visualization script."""
    print("Running Simple Cityscapes Bounding Box Visualization")
    print("=" * 55)
    
    # Check if the script exists
    script_path = "./simple_cityscapes_bbox_viz.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Run the script
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Script completed successfully!")
            return True
        else:
            print(f"❌ Script failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def run_advanced_visualization():
    """Run the advanced visualization script with command line arguments."""
    print("Running Advanced Cityscapes Bounding Box Visualization")
    print("=" * 58)
    
    # Check if the script exists
    script_path = "./visualize_cityscapes_bboxes.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Example command line arguments
    args = [
        "--data_root", "./data",
        "--img_dir", "cityscapes/leftImg8bit/val",
        "--ann_dir", "cityscapes/gtFine/val", 
        "--output_dir", "./advanced_bbox_visualizations",
        "--num_samples", "3",
        "--min_bbox_area", "500"
    ]
    
    try:
        result = subprocess.run([sys.executable, script_path] + args,
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Script completed successfully!")
            return True
        else:
            print(f"❌ Script failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking Dependencies")
    print("=" * 20)
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            missing.append(name)
    
    # Check optional dependencies
    try:
        import scipy
        print("✓ SciPy available (optimal)")
    except ImportError:
        print("⚠ SciPy not available (fallback method will be used)")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install torch torchvision pillow matplotlib numpy scipy")
        return False
    else:
        print("\n✓ All required dependencies available!")
        return True

def check_dataset_structure():
    """Check if the dataset structure is correct."""
    print("Checking Dataset Structure")
    print("=" * 25)
    
    # Default paths (modify these to match your setup)
    data_root = "./data"
    img_dir = "cityscapes/leftImg8bit/val"
    ann_dir = "cityscapes/gtFine/val"
    
    img_path = os.path.join(data_root, img_dir)
    ann_path = os.path.join(data_root, ann_dir)
    
    print(f"Checking image directory: {img_path}")
    if os.path.exists(img_path):
        # Count images
        image_count = 0
        for root, dirs, files in os.walk(img_path):
            image_count += len([f for f in files if f.endswith('_leftImg8bit.png')])
        print(f"✓ Image directory found with {image_count} images")
    else:
        print(f"❌ Image directory not found")
        return False
    
    print(f"Checking annotation directory: {ann_path}")
    if os.path.exists(ann_path):
        # Count annotations
        ann_count = 0
        for root, dirs, files in os.walk(ann_path):
            ann_count += len([f for f in files if f.endswith('_gtFine_labelTrainIds.png')])
        print(f"✓ Annotation directory found with {ann_count} annotations")
    else:
        print(f"❌ Annotation directory not found")
        return False
    
    print("\n✓ Dataset structure looks good!")
    return True

def main():
    """Main function to demonstrate the visualization scripts."""
    print("Cityscapes Bounding Box Visualization - Usage Example")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        return 1
    
    print("\n")
    
    # Check dataset structure
    if not check_dataset_structure():
        print("\n❌ Please check your dataset paths and structure.")
        print("Expected structure:")
        print("  ./data/cityscapes/leftImg8bit/val/<city>/<image>_leftImg8bit.png")
        print("  ./data/cityscapes/gtFine/val/<city>/<image>_gtFine_labelTrainIds.png")
        print("\nModify the paths in the scripts if your structure is different.")
        return 1
    
    print("\n")
    
    # Offer choice of which script to run
    print("Choose which visualization script to run:")
    print("1. Simple script (easier, predefined paths)")
    print("2. Advanced script (more options, command line arguments)")
    print("3. Both scripts")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        run_simple_visualization()
    elif choice == "2":
        run_advanced_visualization()
    elif choice == "3":
        print("\nRunning simple script first...")
        run_simple_visualization()
        print("\n" + "="*60)
        print("Running advanced script...")
        run_advanced_visualization()
    else:
        print("Invalid choice. Running simple script by default...")
        run_simple_visualization()
    
    print("\n" + "="*60)
    print("Usage example complete!")
    print("Check the output directories for generated visualizations:")
    print("  - ./bbox_visualizations/ (simple script)")
    print("  - ./advanced_bbox_visualizations/ (advanced script)")
    
    return 0

if __name__ == "__main__":
    exit(main())
