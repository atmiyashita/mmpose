#!/usr/bin/env python3
"""
Chunked multi-GPU video inference - Split video into chunks, process independently, then merge
Much faster than frame-by-frame multi-GPU processing
"""

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import argparse
import time
import os
import tempfile
import subprocess
from pathlib import Path


class DetectionModel:
    def __init__(self, model_path, num_classes=5, device='cuda:0'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path, num_classes)
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Class mapping for new model (extend strategy)
        self.class_names = {0: "background", 1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi", 4: "kaiko"}

    def _load_model(self, model_path, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def detect(self, image, score_threshold=0.5):
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        img_tensor = self.transform(pil_image).to(self.device)
        
        with torch.no_grad():
            outputs = self.model([img_tensor])
        
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        
        # Filter by score and exclude background
        valid_mask = (scores >= score_threshold) & (labels != 0)
        
        return boxes[valid_mask], scores[valid_mask], labels[valid_mask]


def process_video_chunk(gpu_id, input_chunk_path, output_chunk_path, detection_model_path, 
                       pose_config, pose_checkpoint, detection_threshold, pose_threshold):
    """Process a single video chunk on one GPU"""
    
    # Set GPU for this worker
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    print(f"🔄 GPU {gpu_id}: Processing {input_chunk_path}")
    
    # Register MMPose modules
    register_all_modules()
    
    # Load models on this GPU
    detection_model = DetectionModel(detection_model_path, device=device)
    pose_model = init_model(pose_config, pose_checkpoint, device=device)
    
    print(f"✅ GPU {gpu_id}: Models loaded")
    
    # Open input video chunk
    cap = cv2.VideoCapture(input_chunk_path)
    if not cap.isOpened():
        print(f"❌ GPU {gpu_id}: Could not open {input_chunk_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_chunk_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection
        boxes, scores, labels = detection_model.detect(frame, detection_threshold)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        if len(boxes) > 0:
            detections_count += len(boxes)

            # Draw detection boxes first
            for i, (bbox, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = bbox.astype(int)

                # Draw bounding box with color coding
                if label == 1:
                    color = (0, 255, 0)      # Green for kaiko_live
                elif label == 2:
                    color = (0, 0, 255)      # Red for kaiko_pao1
                elif label == 3:
                    color = (0, 165, 255)    # Orange for kaiko_rhi
                elif label == 4:
                    color = (255, 0, 0)      # Blue for kaiko (your new data)
                else:
                    color = (128, 128, 128)  # Gray for unknown

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Draw class label and confidence with silkworm ID
                class_name = detection_model.class_names.get(label, f"class_{label}")
                label_text = f"ID{i+1} {class_name}: {score:.2f}"
                cv2.putText(vis_frame, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Pose estimation for ALL detections
            pose_results = inference_topdown(pose_model, frame, boxes)

            # Draw pose for each silkworm
            for i, (bbox, score, label) in enumerate(zip(boxes, scores, labels)):
                # Use same color as detection box
                if label == 1:
                    kpt_color = (0, 255, 0)      # Green
                elif label == 2:
                    kpt_color = (0, 0, 255)      # Red
                elif label == 3:
                    kpt_color = (0, 165, 255)    # Orange
                elif label == 4:
                    kpt_color = (255, 0, 0)      # Blue
                else:
                    kpt_color = (128, 128, 128)  # Gray

                # Draw pose keypoints if available
                if i < len(pose_results):
                    pose_result = pose_results[i]
                    keypoints = pose_result.pred_instances.keypoints[0]
                    keypoint_scores = pose_result.pred_instances.keypoint_scores[0]

                    # Draw keypoints with better visibility
                    for j, (kpt, kpt_score) in enumerate(zip(keypoints, keypoint_scores)):
                        if kpt_score > pose_threshold:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(vis_frame, (x, y), 4, kpt_color, -1)
                            cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 2)  # White border
                            cv2.putText(vis_frame, f"{j}", (x+8, y-8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1)

                    # Draw skeleton connections
                    skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
                    for connection in skeleton:
                        if len(keypoints) > max(connection):
                            pt1_idx, pt2_idx = connection
                            if (keypoint_scores[pt1_idx] > pose_threshold and
                                keypoint_scores[pt2_idx] > pose_threshold):
                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                                cv2.line(vis_frame, pt1, pt2, kpt_color, 3)
        
        out.write(vis_frame)
        frame_count += 1
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            print(f"GPU {gpu_id}: {frame_count}/{total_frames} frames, {fps_current:.1f} fps, {detections_count} detections")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    fps_avg = frame_count / elapsed if elapsed > 0 else 0
    print(f"✅ GPU {gpu_id}: Completed {frame_count} frames in {elapsed:.1f}s ({fps_avg:.1f} fps)")
    print(f"   Total detections: {detections_count}")


def split_video(input_path, num_chunks, temp_dir):
    """Split video into chunks using ffmpeg"""
    
    print(f"📹 Splitting video into {num_chunks} chunks...")
    
    # Get video duration
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    chunk_duration = duration / num_chunks
    chunk_paths = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp4")
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-c', 'copy',  # Copy without re-encoding for speed
            chunk_path
        ]
        
        subprocess.run(cmd, capture_output=True)
        chunk_paths.append(chunk_path)
        print(f"   Chunk {i}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s")
    
    print(f"✅ Video split into {len(chunk_paths)} chunks")
    return chunk_paths


def merge_video_chunks(chunk_paths, output_path):
    """Merge processed video chunks using ffmpeg"""
    
    print(f"🔗 Merging {len(chunk_paths)} processed chunks...")
    
    # Create file list for ffmpeg
    list_file = os.path.join(os.path.dirname(chunk_paths[0]), 'filelist.txt')
    with open(list_file, 'w') as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{os.path.basename(chunk_path)}'\n")
    
    # Merge chunks
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', list_file,
        '-c', 'copy',  # Copy without re-encoding
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, cwd=os.path.dirname(chunk_paths[0]))
    
    # Clean up
    os.remove(list_file)
    
    print(f"✅ Video merged: {output_path}")


def chunked_multi_gpu_inference(video_path, output_path, detection_model_path, pose_config, pose_checkpoint,
                               detection_threshold=0.5, pose_threshold=0.3, num_gpus=3):
    """Main function for chunked multi-GPU inference"""
    
    print(f"🚀 Chunked Multi-GPU Video Inference")
    print(f"   Video: {video_path}")
    print(f"   Output: {output_path}")
    print(f"   GPUs: {num_gpus}")
    
    start_time = time.time()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Temp directory: {temp_dir}")
        
        # Step 1: Split video into chunks
        input_chunks = split_video(video_path, num_gpus, temp_dir)
        
        # Step 2: Process chunks in parallel
        output_chunks = []
        processes = []
        
        for i, input_chunk in enumerate(input_chunks):
            output_chunk = os.path.join(temp_dir, f"processed_chunk_{i}.mp4")
            output_chunks.append(output_chunk)
            
            # Start process for this chunk
            process = Process(target=process_video_chunk, args=(
                i, input_chunk, output_chunk, detection_model_path,
                pose_config, pose_checkpoint, detection_threshold, pose_threshold
            ))
            process.start()
            processes.append(process)
        
        print(f"🔄 Started {len(processes)} parallel processes")
        
        # Wait for all processes to complete
        for i, process in enumerate(processes):
            process.join()
            print(f"✅ Process {i} completed")
        
        # Step 3: Merge processed chunks
        merge_video_chunks(output_chunks, output_path)
    
    total_time = time.time() - start_time
    print(f"🎉 Chunked multi-GPU inference completed in {total_time:.1f}s")
    print(f"💾 Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Chunked Multi-GPU Video Inference')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--output', default='output_chunked.mp4', help='Output video path')
    parser.add_argument('--detection-model', 
                       default='work_dirs/detection_training_extend/final_model.pth',
                       help='Detection model path')
    parser.add_argument('--pose-config',
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-11keypoints-expanded.py',
                       help='Pose config path')
    parser.add_argument('--pose-checkpoint',
                       default='work_dirs/hrnet_silkworm_11keypoints_expanded/best_coco_AP_epoch_200.pth',
                       help='Pose checkpoint path')
    parser.add_argument('--detection-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--pose-threshold', type=float, default=0.3,
                       help='Pose keypoint confidence threshold')
    parser.add_argument('--num-gpus', type=int, default=3,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    chunked_multi_gpu_inference(
        args.video, args.output, args.detection_model,
        args.pose_config, args.pose_checkpoint,
        args.detection_threshold, args.pose_threshold, args.num_gpus
    )


if __name__ == '__main__':
    main()
