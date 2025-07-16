#!/usr/bin/env python3
"""
Convert 11-keypoint dataset to 4-keypoint dataset (H1, T1, A2, A9)
"""

import os
import json
import shutil
import argparse
import numpy as np


def convert_keypoints_to_4point(keypoints_11, num_keypoints_11):
    """Convert 11-keypoint format to 4-keypoint format"""
    
    # Original 11-keypoint mapping:
    # 0:H1, 1:T1, 2:T2, 3:T3, 4:A2, 5:A3, 6:A4, 7:A5, 8:A6, 9:A8, 10:A9
    
    # New 4-keypoint mapping:
    # 0:H1, 1:T1, 2:A2, 3:A9
    
    # Extract the 4 keypoints we want (indices: 0, 1, 4, 10)
    selected_indices = [0, 1, 4, 10]  # H1, T1, A2, A9
    
    keypoints_4 = []
    visible_count = 0
    
    for new_idx, old_idx in enumerate(selected_indices):
        # Each keypoint has 3 values: x, y, visibility
        x = keypoints_11[old_idx * 3]
        y = keypoints_11[old_idx * 3 + 1] 
        v = keypoints_11[old_idx * 3 + 2]
        
        keypoints_4.extend([x, y, v])
        
        if v > 0:  # visible
            visible_count += 1
    
    return keypoints_4, visible_count


def convert_dataset(input_dir, output_dir):
    """Convert 11-keypoint dataset to 4-keypoint dataset"""
    
    print("🔄 Converting 11-keypoint dataset to 4-keypoint dataset...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Copy images (no change needed)
    print("📁 Copying images...")
    src_images = os.path.join(input_dir, 'images')
    dst_images = os.path.join(output_dir, 'images')
    if os.path.exists(dst_images):
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)
    
    # Process each annotation file
    for split in ['train', 'val', 'merged']:
        ann_file = os.path.join(input_dir, f'annotations/person_keypoints_{split}.json')
        if not os.path.exists(ann_file):
            continue
            
        print(f"🔄 Converting {split} annotations...")
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Update categories for 4 keypoints
        new_categories = [{
            'supercategory': 'silkworm',
            'id': 1,
            'name': 'silkworm',
            'keypoints': ['H1', 'T1', 'A2', 'A9'],
            'skeleton': [[1, 2], [2, 3], [3, 4]]  # 1-indexed for COCO format
        }]
        
        # Convert annotations
        new_annotations = []
        converted_count = 0
        
        for ann in data['annotations']:
            if 'keypoints' not in ann:
                continue
                
            # Convert keypoints from 11 to 4
            keypoints_4, num_visible = convert_keypoints_to_4point(
                ann['keypoints'], ann['num_keypoints'])
            
            # Only keep annotations with at least 2 visible keypoints
            if num_visible >= 2:
                new_ann = ann.copy()
                new_ann['keypoints'] = keypoints_4
                new_ann['num_keypoints'] = num_visible
                new_annotations.append(new_ann)
                converted_count += 1
        
        # Create new dataset
        new_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': new_categories,
            'images': data['images'],
            'annotations': new_annotations
        }
        
        # Save converted dataset
        output_file = os.path.join(output_dir, f'annotations/person_keypoints_{split}.json')
        with open(output_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"   {split}: {len(data['annotations'])} → {converted_count} annotations")
    
    print(f"✅ Conversion completed!")
    print(f"📁 Output: {output_dir}")
    
    return output_dir


def analyze_4keypoint_quality(dataset_dir):
    """Analyze the quality of 4-keypoint dataset"""
    
    print(f"\n📊 4-Keypoint Dataset Analysis:")
    print("=" * 50)
    
    train_file = os.path.join(dataset_dir, 'annotations/person_keypoints_train.json')
    if not os.path.exists(train_file):
        print("❌ Training file not found")
        return
    
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    keypoint_names = ['H1', 'T1', 'A2', 'A9']
    visibility_counts = [0, 0, 0, 0]
    total_annotations = len(data['annotations'])
    
    for ann in data['annotations']:
        keypoints = ann['keypoints']
        for i in range(4):  # 4 keypoints
            visibility = keypoints[i * 3 + 2]  # visibility flag
            if visibility > 0:
                visibility_counts[i] += 1
    
    print("Keypoint visibility rates:")
    for i, name in enumerate(keypoint_names):
        rate = visibility_counts[i] / total_annotations if total_annotations > 0 else 0
        print(f"  {name}: {visibility_counts[i]:3d}/{total_annotations:3d} ({rate:.1%})")
    
    # Keypoints per annotation
    keypoints_per_ann = [ann['num_keypoints'] for ann in data['annotations']]
    print(f"\nKeypoints per annotation:")
    print(f"  Average: {np.mean(keypoints_per_ann):.1f}")
    print(f"  Min: {min(keypoints_per_ann)}")
    print(f"  Max: {max(keypoints_per_ann)}")
    
    print(f"\nTotal training annotations: {total_annotations}")


def main():
    parser = argparse.ArgumentParser(description='Convert to 4-keypoint dataset')
    parser.add_argument('--input-dir',
                       default='../data/silkworm/keypoints_merged',
                       help='Input 11-keypoint dataset directory')
    parser.add_argument('--output-dir',
                       default='../data/silkworm/keypoints_4point',
                       help='Output 4-keypoint dataset directory')
    
    args = parser.parse_args()
    
    print("🚀 Converting to 4-Keypoint Dataset")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Keypoints: H1 (head), T1 (thorax), A2 (abdomen), A9 (tail)")
    
    # Convert dataset
    output_dir = convert_dataset(args.input_dir, args.output_dir)
    
    # Analyze quality
    analyze_4keypoint_quality(output_dir)
    
    print(f"\n✅ 4-keypoint dataset ready for training!")


if __name__ == '__main__':
    main()
