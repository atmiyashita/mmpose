#!/usr/bin/env python3
"""
Prepare keypoint training data by merging part1 and part2 datasets
"""

import os
import json
import shutil
import argparse
from collections import defaultdict
import numpy as np


def merge_datasets(part1_dir, part2_dir, output_dir):
    """Merge two COCO keypoint datasets"""
    
    print("🔄 Merging keypoint datasets...")
    
    # Load both datasets
    with open(os.path.join(part1_dir, 'annotations/person_keypoints_default.json'), 'r') as f:
        data1 = json.load(f)
    
    with open(os.path.join(part2_dir, 'annotations/person_keypoints_default.json'), 'r') as f:
        data2 = json.load(f)
    
    print(f"Part 1: {len(data1['images'])} images, {len(data1['annotations'])} annotations")
    print(f"Part 2: {len(data2['images'])} images, {len(data2['annotations'])} annotations")
    
    # Create merged dataset
    merged_data = {
        'info': data1.get('info', {}),
        'licenses': data1.get('licenses', []),
        'categories': data1['categories'],  # Should be the same
        'images': [],
        'annotations': []
    }
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Track ID mappings
    image_id_offset = 0
    annotation_id_offset = 0
    
    # Process part 1
    print("Processing part 1...")
    for img in data1['images']:
        # Copy image file
        src_path = os.path.join(part1_dir, 'images/default', img['file_name'])
        dst_path = os.path.join(output_dir, 'images', f"part1_{img['file_name']}")
        shutil.copy2(src_path, dst_path)
        
        # Update image info
        img_copy = img.copy()
        img_copy['file_name'] = f"part1_{img['file_name']}"
        merged_data['images'].append(img_copy)
    
    # Add part 1 annotations
    for ann in data1['annotations']:
        merged_data['annotations'].append(ann)
    
    # Update offsets for part 2
    image_id_offset = max([img['id'] for img in data1['images']]) + 1
    annotation_id_offset = max([ann['id'] for ann in data1['annotations']]) + 1
    
    # Process part 2
    print("Processing part 2...")
    for img in data2['images']:
        # Copy image file
        src_path = os.path.join(part2_dir, 'images/default', img['file_name'])
        dst_path = os.path.join(output_dir, 'images', f"part2_{img['file_name']}")
        shutil.copy2(src_path, dst_path)
        
        # Update image info with new ID
        img_copy = img.copy()
        img_copy['id'] = img['id'] + image_id_offset
        img_copy['file_name'] = f"part2_{img['file_name']}"
        merged_data['images'].append(img_copy)
    
    # Add part 2 annotations with updated IDs
    for ann in data2['annotations']:
        ann_copy = ann.copy()
        ann_copy['id'] = ann['id'] + annotation_id_offset
        ann_copy['image_id'] = ann['image_id'] + image_id_offset
        merged_data['annotations'].append(ann_copy)
    
    # Save merged dataset
    merged_file = os.path.join(output_dir, 'annotations/person_keypoints_merged.json')
    with open(merged_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✅ Merged dataset saved:")
    print(f"   Images: {len(merged_data['images'])}")
    print(f"   Annotations: {len(merged_data['annotations'])}")
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


def analyze_keypoint_quality(merged_data):
    """Analyze keypoint annotation quality"""
    
    print(f"\n📊 Keypoint Quality Analysis:")
    print("=" * 50)
    
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
            print(f"  {kpt_name:3s}: {visibility_counts[kpt_name]:3d}/{total_keypoints[kpt_name]:3d} ({rate:.1%})")
    
    # Annotation statistics
    num_keypoints_per_ann = [ann['num_keypoints'] for ann in merged_data['annotations']]
    print(f"\nKeypoints per annotation:")
    print(f"  Average: {np.mean(num_keypoints_per_ann):.1f}")
    print(f"  Min: {min(num_keypoints_per_ann)}")
    print(f"  Max: {max(num_keypoints_per_ann)}")
    
    # Bounding box statistics
    bbox_areas = [ann['area'] for ann in merged_data['annotations']]
    print(f"\nBounding box areas:")
    print(f"  Average: {np.mean(bbox_areas):.1f}")
    print(f"  Min: {min(bbox_areas):.1f}")
    print(f"  Max: {max(bbox_areas):.1f}")


def update_pose_config(output_dir):
    """Update the pose estimation config to use new dataset"""
    
    config_updates = f"""
# Updated configuration for keypoint training
# Dataset: {output_dir}

# Update these paths in your pose config:
data_root = '{output_dir}/'

# Training dataloader
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='images/'),
    )
)

# Validation dataloader  
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_val.json',
        data_prefix=dict(img='images/'),
    )
)

# Evaluator
val_evaluator = dict(
    ann_file='{output_dir}/annotations/person_keypoints_val.json'
)
"""
    
    config_file = os.path.join(output_dir, 'config_updates.py')
    with open(config_file, 'w') as f:
        f.write(config_updates)
    
    print(f"\n📋 Config updates saved: {config_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Keypoint Training Data')
    parser.add_argument('--part1-dir', 
                       default='../data/silkworm/keypoints_traindata/part1',
                       help='Part 1 dataset directory')
    parser.add_argument('--part2-dir',
                       default='../data/silkworm/keypoints_traindata/part2', 
                       help='Part 2 dataset directory')
    parser.add_argument('--output-dir',
                       default='../data/silkworm/keypoints_merged',
                       help='Output directory for merged dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    print("🚀 Preparing Keypoint Training Data")
    print("=" * 60)
    
    # Merge datasets
    merged_data = merge_datasets(args.part1_dir, args.part2_dir, args.output_dir)
    
    # Create train/val split
    train_size, val_size = create_train_val_split(merged_data, args.output_dir, args.val_ratio)
    
    # Analyze quality
    analyze_keypoint_quality(merged_data)
    
    # Update config
    update_pose_config(args.output_dir)
    
    print(f"\n✅ Keypoint training data preparation completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Ready for training with {len(merged_data['annotations'])} keypoint annotations!")


if __name__ == '__main__':
    main()
