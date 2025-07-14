#!/usr/bin/env python3
"""
Improved inference script for silkworm pose estimation
"""

import os
import cv2
import numpy as np
import argparse
from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


def main():
    parser = argparse.ArgumentParser(description='Silkworm Pose Inference')
    parser.add_argument('--config', 
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py',
                       help='Config file path')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/hrnet_silkworm11/epoch_210.pth',
                       help='Checkpoint file path')
    parser.add_argument('--input', '-i', required=True,
                       help='Input image path')
    parser.add_argument('--output', '-o', 
                       help='Output image path (default: input_pose.jpg)')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--kpt-thr', type=float, default=0.1,
                       help='Keypoint confidence threshold')
    parser.add_argument('--show-kpt-idx', action='store_true',
                       help='Show keypoint indices')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found!")
        return
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_pose.jpg"
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Initialize model
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and process image
    print(f"Processing image: {args.input}")
    img_array = cv2.imread(args.input)
    if img_array is None:
        print(f"Error: Could not load image '{args.input}'")
        return
        
    h, w = img_array.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Create bounding box for the entire image
    person_results = [np.array([0, 0, w, h], dtype=np.float32)]
    
    # Run inference
    print("Running inference...")
    try:
        preds = inference_topdown(model, args.input, person_results)
        print(f"Inference completed! Found {len(preds)} predictions")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    if len(preds) == 0:
        print("No pose predictions found!")
        return
    
    # Print keypoint information
    pred_instances = preds[0].pred_instances
    keypoints = pred_instances.keypoints[0]  # First (and only) instance
    scores = pred_instances.keypoint_scores[0]
    
    print(f"\nKeypoint predictions (confidence > {args.kpt_thr}):")
    keypoint_names = ['H1', 'T1', 'T2', 'T3', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9']
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        if score > args.kpt_thr:
            print(f"  {i:2d} {keypoint_names[i]:3s}: ({kpt[0]:6.1f}, {kpt[1]:6.1f}) conf={score:.3f}")
    
    # Initialize visualizer
    print("Creating visualization...")
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(model.dataset_meta)
    
    # Visualize results
    visualizer.add_datasample(
        'silkworm_pose',
        img_array,
        data_sample=preds[0],
        draw_gt=False,
        draw_heatmap=False,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style='mmpose',
        kpt_thr=args.kpt_thr
    )
    
    # Save result
    vis_result = visualizer.get_image()
    cv2.imwrite(args.output, vis_result)
    print(f"Result saved to: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
