#!/usr/bin/env python3
"""
Video inference script for silkworm pose estimation
Processes MP4 files frame by frame and outputs annotated video
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


def process_video(model, input_path, output_path, kpt_thr=0.1, show_kpt_idx=False):
    """Process video file and generate pose annotations"""
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize visualizer
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(model.dataset_meta)
    
    # Create bounding box for the entire frame
    person_results = [np.array([0, 0, width, height], dtype=np.float32)]
    
    frame_count = 0
    keypoint_names = ['H1', 'T1', 'T2', 'T3', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9']
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Run inference on current frame
                preds = inference_topdown(model, frame, person_results)
                
                if len(preds) > 0:
                    # Clear previous visualization
                    visualizer.set_image(frame)
                    
                    # Add pose visualization
                    visualizer.add_datasample(
                        f'frame_{frame_count}',
                        frame,
                        data_sample=preds[0],
                        draw_gt=False,
                        draw_heatmap=False,
                        show_kpt_idx=show_kpt_idx,
                        skeleton_style='mmpose',
                        kpt_thr=kpt_thr
                    )
                    
                    # Get visualization result
                    vis_frame = visualizer.get_image()
                    
                    # Optional: Print keypoint info for first few frames
                    if frame_count < 5:
                        pred_instances = preds[0].pred_instances
                        keypoints = pred_instances.keypoints[0]
                        scores = pred_instances.keypoint_scores[0]
                        
                        print(f"\nFrame {frame_count} keypoints (conf > {kpt_thr}):")
                        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                            if score > kpt_thr:
                                print(f"  {keypoint_names[i]:3s}: ({kpt[0]:6.1f}, {kpt[1]:6.1f}) conf={score:.3f}")
                else:
                    # No pose detected, use original frame
                    vis_frame = frame
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                vis_frame = frame
            
            # Write frame to output video
            out.write(vis_frame)
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Processed {frame_count} frames")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description='Silkworm Pose Video Inference')
    parser.add_argument('--config', 
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py',
                       help='Config file path')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/hrnet_silkworm11/epoch_210.pth',
                       help='Checkpoint file path')
    parser.add_argument('--input', '-i', required=True,
                       help='Input video path (MP4)')
    parser.add_argument('--output', '-o', 
                       help='Output video path (default: input_pose.mp4)')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--kpt-thr', type=float, default=0.1,
                       help='Keypoint confidence threshold')
    parser.add_argument('--show-kpt-idx', action='store_true',
                       help='Show keypoint indices')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every N frames (1=all frames, 2=every other frame, etc.)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_pose.mp4"
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Initialize model
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process video
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    try:
        frame_count = process_video(
            model, 
            args.input, 
            args.output, 
            kpt_thr=args.kpt_thr,
            show_kpt_idx=args.show_kpt_idx
        )
        print(f"\n✅ Video processing completed!")
        print(f"📹 Processed {frame_count} frames")
        print(f"💾 Output saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")


if __name__ == '__main__':
    main()
