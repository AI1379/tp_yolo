"""Fix YOLO dataset structure and label format."""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def fix_label_format(label_path: Path, output_path: Path) -> bool:
    """Fix label format from comma-separated to space-separated coordinates.
    
    Args:
        label_path: Input label file path
        output_path: Output label file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            return False
        
        # Split by line (in case there are multiple objects, though unlikely here)
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split(' ', 1)  # Split into class_id and coordinates
            if len(parts) != 2:
                return False
                
            class_id = parts[0]
            coords_str = parts[1]
            
            # Split coordinates by space, then replace commas with spaces
            coord_pairs = coords_str.split()
            fixed_coords = []
            
            for pair in coord_pairs:
                if ',' in pair:
                    x, y = pair.split(',')
                    fixed_coords.extend([x, y])
                else:
                    # Already in correct format?
                    fixed_coords.append(pair)
            
            fixed_line = f"{class_id} {' '.join(fixed_coords)}"
            fixed_lines.append(fixed_line)
        
        # Write fixed content
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(fixed_lines))
        
        return True
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False


def flatten_dataset(source_base: Path, dest_base: Path, fix_labels: bool = False):
    """Flatten dataset structure and optionally fix label format.
    
    Args:
        source_base: Source directory containing Part subdirectories
        dest_base: Destination directory for flattened structure
        fix_labels: Whether to fix label format (for segmentation)
    """
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        src_images = source_base / split / 'images'
        src_labels = source_base / split / 'labels'
        dst_images = dest_base / split / 'images'
        dst_labels = dest_base / split / 'labels'
        
        # Create destination directories
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        if not src_images.exists():
            print(f"  Skipping {split} - source directory not found")
            continue
        
        # Get all Part directories
        part_dirs = sorted([d for d in src_images.iterdir() if d.is_dir() and d.name.startswith('Part')])
        
        if not part_dirs:
            print(f"  No Part directories found in {src_images}")
            continue
        
        # Process each Part directory
        total_images = 0
        total_labels = 0
        
        for part_dir in tqdm(part_dirs, desc=f"  {split}"):
            # Copy images
            for img_file in part_dir.glob('*.jpg'):
                dst_img = dst_images / img_file.name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                    total_images += 1
            
            # Copy/fix labels
            label_part_dir = src_labels / part_dir.name
            if label_part_dir.exists():
                for label_file in label_part_dir.glob('*.txt'):
                    dst_label = dst_labels / label_file.name
                    
                    if fix_labels:
                        if fix_label_format(label_file, dst_label):
                            total_labels += 1
                    else:
                        if not dst_label.exists():
                            shutil.copy2(label_file, dst_label)
                            total_labels += 1
        
        print(f"  Copied {total_images} images and {total_labels} labels")


def main():
    base_path = Path(__file__).parent / 'data' / 'TP-Dataset' / 'YOLO_Data'
    
    # Fix boxes dataset (detection)
    print("=" * 60)
    print("Fixing boxes dataset (detection)")
    print("=" * 60)
    boxes_src = base_path / 'boxes'
    boxes_dst = base_path / 'boxes_fixed'
    
    if boxes_src.exists():
        flatten_dataset(boxes_src, boxes_dst, fix_labels=False)
        print(f"\n✓ Boxes dataset fixed and saved to {boxes_dst}")
    else:
        print(f"Boxes source not found: {boxes_src}")
    
    # Fix contours dataset (segmentation)
    print("\n" + "=" * 60)
    print("Fixing contours dataset (segmentation)")
    print("=" * 60)
    contours_src = base_path / 'contours'
    contours_dst = base_path / 'contours_fixed'
    
    if contours_src.exists():
        flatten_dataset(contours_src, contours_dst, fix_labels=True)
        print(f"\n✓ Contours dataset fixed and saved to {contours_dst}")
    else:
        print(f"Contours source not found: {contours_src}")
    
    print("\n" + "=" * 60)
    print("Dataset fixing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update your YAML files to point to the *_fixed directories")
    print("2. Remove or rename the old directories if everything works")
    print("3. DO NOT use check_dataset() - it's for HUB upload only")
    print("4. Use model.train() directly for training")


if __name__ == '__main__':
    main()
