#!/usr/bin/env python3
"""
Prepare expanded keypoint training data by merging all kaiko datasets
"""

import os
import json
import shutil
import argparse
from collections import defaultdict
import numpy as np
import glob


def merge_all_datasets(input_base_dir, output_dir):
    """Merge all COCO keypoint datasets"""
    
    print("🔄 Merging expanded keypoint datasets...")
    
    # Find all dataset directories
    dataset_dirs = []
    for item in os.listdir(input_base_dir):
        item_path = os.path.join(input_base_dir, item)
        if os.path.isdir(item_path):
            ann_file = None
            # Check for annotation file
            ann_dir = os.path.join(item_path, 'annotations')
            if os.path.exists(ann_dir):
                json_files = glob.glob(os.path.join(ann_dir, '*.json'))
                if json_files:
                    ann_file = json_files[0]  # Take first JSON file
            
            if ann_file:
                dataset_dirs.append((item, item_path, ann_file))
    
    print(f"Found {len(dataset_dirs)} datasets:")
    for name, path, ann_file in dataset_dirs:
        print(f"  - {name}: {ann_file}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Initialize merged dataset
    merged_data = {
        'info': {'description': 'Merged Silkworm Keypoint Dataset'},
        'licenses': [],
        'categories': [],
        'images': [],
        'annotations': []
    }
    
    # Track ID mappings
    image_id_offset = 0
    annotation_id_offset = 0
    total_images = 0
    total_annotations = 0
    
    # Process each dataset
    for dataset_name, dataset_path, ann_file in dataset_dirs:
        print(f"\n🔄 Processing {dataset_name}...")
        
        # Load dataset
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        print(f"   Images: {len(data['images'])}, Annotations: {len(data['annotations'])}")
        
        # Set categories from first dataset
        if not merged_data['categories']:
            merged_data['categories'] = data['categories']
        
        # Find images directory
        img_dir = os.path.join(dataset_path, 'images')
        if not os.path.exists(img_dir):
            # Try subdirectories
            subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d != 'annotations']
            if subdirs:
                img_dir = os.path.join(dataset_path, subdirs[0])
                if 'images' in subdirs:
                    img_dir = os.path.join(dataset_path, 'images')
                elif 'default' in subdirs:
                    img_dir = os.path.join(dataset_path, 'images', 'default')
        
        # Process images
        dataset_images = 0
        for img in data['images']:
            # Find source image file
            src_path = None
            possible_paths = [
                os.path.join(img_dir, img['file_name']),
                os.path.join(dataset_path, 'images', 'default', img['file_name']),
                os.path.join(dataset_path, 'images', img['file_name'])
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    src_path = path
                    break
            
            if src_path:
                # Copy image file with unique name
                new_filename = f"{dataset_name}_{img['file_name']}"
                dst_path = os.path.join(output_dir, 'images', new_filename)
                shutil.copy2(src_path, dst_path)
                
                # Update image info
                img_copy = img.copy()
                img_copy['id'] = img['id'] + image_id_offset
                img_copy['file_name'] = new_filename
                merged_data['images'].append(img_copy)
                dataset_images += 1
            else:
                print(f"   Warning: Image not found: {img['file_name']}")
        
        # Process annotations
        dataset_annotations = 0
        for ann in data['annotations']:
            if 'keypoints' in ann:
                ann_copy = ann.copy()
                ann_copy['id'] = ann['id'] + annotation_id_offset
                ann_copy['image_id'] = ann['image_id'] + image_id_offset
                merged_data['annotations'].append(ann_copy)
                dataset_annotations += 1
        
        # Update offsets for next dataset
        if data['images']:
            image_id_offset = max([img['id'] + image_id_offset for img in data['images']]) + 1
        if data['annotations']:
            annotation_id_offset = max([ann['id'] + annotation_id_offset for ann in data['annotations']]) + 1
        
        total_images += dataset_images
        total_annotations += dataset_annotations
        print(f"   Processed: {dataset_images} images, {dataset_annotations} annotations")
    
    # Save merged dataset
    merged_file = os.path.join(output_dir, 'annotations/person_keypoints_merged.json')
    with open(merged_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✅ Merged dataset completed!")
    print(f"   Total images: {total_images}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Output: {merged_file}")
    
    return merged_data


def create_train_val_split(merged_data, output_dir, val_ratio=0.2, seed=42):
    """Create train/validation split"""
    
    print(f"\n📂 Creating train/val split (val_ratio={val_ratio})...")
    
    np.random.seed(seed)
    
    # Get all image IDs that have annotations
    annotated_image_ids = set(ann['image_id'] for ann in merged_data['annotations'])
    annotated_image_ids = list(annotated_image_ids)
    np.random.shuffle(annotated_image_ids)
    
    # Split
    val_size = int(len(annotated_image_ids) * val_ratio)
    val_image_ids = set(annotated_image_ids[:val_size])
    train_image_ids = set(annotated_image_ids[val_size:])
    
    print(f"Training images: {len(train_image_ids)}")
    print(f"Validation images: {len(val_image_ids)}")
    
    # Create train and val datasets
    for split_name, split_ids in [('train', train_image_ids), ('val', val_image_ids)]:
        # Filter images
        split_images = [img for img in merged_data['images'] if img['id'] in split_ids]
        
        # Filter annotations
        split_annotations = [ann for ann in merged_data['annotations'] if ann['image_id'] in split_ids]
        
        # Create split dataset
        split_data = {
            'info': merged_data.get('info', {}),
            'licenses': merged_data.get('licenses', []),
            'categories': merged_data['categories'],
            'images': split_images,
            'annotations': split_annotations
        }
        
        # Save
        split_file = os.path.join(output_dir, f'annotations/person_keypoints_{split_name}.json')
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"   {split_name.capitalize()}: {len(split_images)} images, {len(split_annotations)} annotations")
        print(f"   Saved: {split_file}")
    
    return len(train_image_ids), len(val_image_ids)


def analyze_expanded_dataset(merged_data):
    """Analyze the expanded dataset quality"""
    
    print(f"\n📊 Expanded Dataset Analysis:")
    print("=" * 60)
    
    # Keypoint visibility statistics
    keypoint_names = merged_data['categories'][0]['keypoints']
    visibility_counts = defaultdict(int)
    total_keypoints = defaultdict(int)
    
    for ann in merged_data['annotations']:
        keypoints = ann['keypoints']
        num_keypoints = len(keypoints) // 3  # x, y, visibility for each keypoint
        
        for i in range(num_keypoints):
            kpt_name = keypoint_names[i]
            visibility = keypoints[i * 3 + 2]  # visibility flag
            total_keypoints[kpt_name] += 1
            if visibility > 0:  # visible
                visibility_counts[kpt_name] += 1
    
    print("Keypoint visibility rates:")
    for kpt_name in keypoint_names:
        if total_keypoints[kpt_name] > 0:
            rate = visibility_counts[kpt_name] / total_keypoints[kpt_name]
            print(f"  {kpt_name:3s}: {visibility_counts[kpt_name]:4d}/{total_keypoints[kpt_name]:4d} ({rate:.1%})")
    
    # Annotation statistics
    num_keypoints_per_ann = [ann['num_keypoints'] for ann in merged_data['annotations']]
    print(f"\nKeypoints per annotation:")
    print(f"  Average: {np.mean(num_keypoints_per_ann):.1f}")
    print(f"  Min: {min(num_keypoints_per_ann)}")
    print(f"  Max: {max(num_keypoints_per_ann)}")
    
    print(f"\nDataset size:")
    print(f"  Total images: {len(merged_data['images'])}")
    print(f"  Total annotations: {len(merged_data['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Expanded Keypoint Training Data')
    parser.add_argument('--input-dir',
                       default='../data/silkworm/keypoints_traindata',
                       help='Input directory containing all datasets')
    parser.add_argument('--output-dir',
                       default='../data/silkworm/keypoints_expanded',
                       help='Output directory for merged dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    print("🚀 Preparing Expanded Keypoint Training Data")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Merge all datasets
    merged_data = merge_all_datasets(args.input_dir, args.output_dir)
    
    # Create train/val split
    train_size, val_size = create_train_val_split(merged_data, args.output_dir, args.val_ratio)
    
    # Analyze quality
    analyze_expanded_dataset(merged_data)
    
    print(f"\n✅ Expanded keypoint dataset ready for training!")
    print(f"📈 Significant improvement: {len(merged_data['annotations'])} annotations vs previous 118!")


if __name__ == '__main__':
    main()
