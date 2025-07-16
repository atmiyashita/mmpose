#!/usr/bin/env python3
"""
Data preparation and analysis script for silkworm detection training
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def analyze_dataset(annotation_file, image_root):
    """Analyze the dataset and provide statistics"""
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print("📊 Dataset Analysis")
    print("=" * 50)
    
    # Basic statistics
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data['categories'])
    
    print(f"Images: {num_images}")
    print(f"Annotations: {num_annotations}")
    print(f"Categories: {num_categories}")
    print(f"Annotations per image: {num_annotations/num_images:.2f}")
    
    # Category information
    print("\n📋 Categories:")
    for cat in data['categories']:
        print(f"  ID {cat['id']}: {cat['name']}")
    
    # Image size analysis
    image_sizes = []
    for img_info in data['images']:
        image_sizes.append((img_info['width'], img_info['height']))
    
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    print(f"\n📐 Image Sizes:")
    print(f"  Width: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")
    print(f"  Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")
    
    # Bounding box analysis
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    
    for ann in data['annotations']:
        bbox = ann['bbox']  # [x, y, width, height]
        w, h = bbox[2], bbox[3]
        area = w * h
        
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_areas.append(area)
    
    print(f"\n📦 Bounding Box Statistics:")
    print(f"  Width: {min(bbox_widths):.1f} - {max(bbox_widths):.1f} (avg: {np.mean(bbox_widths):.1f})")
    print(f"  Height: {min(bbox_heights):.1f} - {max(bbox_heights):.1f} (avg: {np.mean(bbox_heights):.1f})")
    print(f"  Area: {min(bbox_areas):.1f} - {max(bbox_areas):.1f} (avg: {np.mean(bbox_areas):.1f})")
    
    # Check for missing images
    missing_images = []
    for img_info in data['images']:
        img_path = os.path.join(image_root, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"\n⚠️ Missing Images ({len(missing_images)}):")
        for img in missing_images[:5]:  # Show first 5
            print(f"  {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    else:
        print(f"\n✅ All images found!")
    
    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'bbox_areas': bbox_areas,
        'bbox_widths': bbox_widths,
        'bbox_heights': bbox_heights,
        'image_sizes': image_sizes
    }


def create_train_val_split(annotation_file, output_dir, val_ratio=0.2, seed=42):
    """Split dataset into train and validation sets"""
    
    np.random.seed(seed)
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in data['images']]
    np.random.shuffle(image_ids)
    
    # Split
    val_size = int(len(image_ids) * val_ratio)
    val_image_ids = set(image_ids[:val_size])
    train_image_ids = set(image_ids[val_size:])
    
    print(f"📂 Creating train/val split:")
    print(f"  Training: {len(train_image_ids)} images")
    print(f"  Validation: {len(val_image_ids)} images")
    
    # Create train and val datasets
    for split_name, split_ids in [('train', train_image_ids), ('val', val_image_ids)]:
        # Filter images
        split_images = [img for img in data['images'] if img['id'] in split_ids]
        
        # Filter annotations
        split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in split_ids]
        
        # Create new dataset
        split_data = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': data['categories'],
            'info': data.get('info', {}),
            'licenses': data.get('licenses', [])
        }
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'instances_{split_name}.json')
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"  💾 {split_name.capitalize()} set saved: {output_file}")
    
    return len(train_image_ids), len(val_image_ids)


def visualize_samples(annotation_file, image_root, output_dir, num_samples=6):
    """Visualize some sample images with annotations"""
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Get image info
    image_info = {img['id']: img for img in data['images']}
    
    # Select random samples
    image_ids = list(annotations_by_image.keys())
    np.random.shuffle(image_ids)
    sample_ids = image_ids[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, image_id in enumerate(sample_ids):
        if i >= len(axes):
            break
            
        # Load image
        img_info = image_info[image_id]
        img_path = os.path.join(image_root, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path)
        axes[i].imshow(img)
        
        # Draw bounding boxes
        for ann in annotations_by_image[image_id]:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[i].add_patch(rect)
            
            # Add label
            axes[i].text(x, y-5, f"kaiko", color='red', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        axes[i].set_title(f"Image {image_id}: {img_info['file_name']}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_ids), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(output_dir, 'sample_annotations.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Sample visualization saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Detection Dataset')
    parser.add_argument('--annotation-file', 
                       default='kaiko_labeled/annotations/instances_default.json',
                       help='COCO annotation file')
    parser.add_argument('--image-root', 
                       default='kaiko_labeled/images/default',
                       help='Root directory of images')
    parser.add_argument('--output-dir', 
                       default='work_dirs/detection_data',
                       help='Output directory for processed data')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--create-split', action='store_true',
                       help='Create train/val split')
    parser.add_argument('--visualize', action='store_true',
                       help='Create sample visualizations')
    
    args = parser.parse_args()
    
    print("🔍 Analyzing dataset...")
    stats = analyze_dataset(args.annotation_file, args.image_root)
    
    if args.create_split:
        print("\n📂 Creating train/val split...")
        train_size, val_size = create_train_val_split(
            args.annotation_file, 
            args.output_dir, 
            args.val_ratio
        )
    
    if args.visualize:
        print("\n🎨 Creating visualizations...")
        visualize_samples(args.annotation_file, args.image_root, args.output_dir)
    
    print(f"\n✅ Data preparation completed!")
    if args.create_split or args.visualize:
        print(f"📁 Outputs saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
