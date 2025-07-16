#!/usr/bin/env python3
"""
Multi-GPU video inference for detection and pose estimation
"""

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import argparse
import time
from tqdm import tqdm
import os


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


def worker_process(gpu_id, frame_queue, result_queue, detection_model_path, pose_config, pose_checkpoint, 
                  detection_threshold, pose_threshold):
    """Worker process for each GPU"""
    
    # Set GPU for this worker
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    # Register MMPose modules
    register_all_modules()
    
    # Load models on this GPU
    detection_model = DetectionModel(detection_model_path, device=device)
    pose_model = init_model(pose_config, pose_checkpoint, device=device)
    
    print(f"Worker {gpu_id}: Models loaded on {device}")
    
    while True:
        try:
            # Get frame from queue
            item = frame_queue.get(timeout=1)
            if item is None:  # Poison pill
                break
                
            frame_idx, frame = item
            
            # Process frame
            processed_frame = process_single_frame(
                frame, detection_model, pose_model, 
                detection_threshold, pose_threshold, gpu_id
            )
            
            # Put result back
            result_queue.put((frame_idx, processed_frame))
            
        except Exception as e:
            print(f"Worker {gpu_id} error: {e}")
            continue
    
    print(f"Worker {gpu_id}: Finished")


def process_single_frame(frame, detection_model, pose_model, detection_threshold, pose_threshold, gpu_id):
    """Process a single frame"""
    
    # Detection
    boxes, scores, labels = detection_model.detect(frame, detection_threshold)
    
    # Create visualization frame
    vis_frame = frame.copy()
    
    if len(boxes) > 0:
        # Pose estimation for each detection
        pose_results = inference_topdown(pose_model, frame, boxes)
        
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
            
            # Draw class label and confidence
            class_name = detection_model.class_names.get(label, f"class_{label}")
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(vis_frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw pose keypoints if available
            if i < len(pose_results):
                pose_result = pose_results[i]
                keypoints = pose_result.pred_instances.keypoints[0]
                keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
                
                # Draw keypoints
                for j, (kpt, kpt_score) in enumerate(zip(keypoints, keypoint_scores)):
                    if kpt_score > pose_threshold:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(vis_frame, (x, y), 3, (0, 255, 255), -1)
                        cv2.putText(vis_frame, f"{j}", (x+5, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                
                # Draw skeleton connections
                skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
                for connection in skeleton:
                    if len(keypoints) > max(connection):
                        pt1_idx, pt2_idx = connection
                        if (keypoint_scores[pt1_idx] > pose_threshold and 
                            keypoint_scores[pt2_idx] > pose_threshold):
                            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                            cv2.line(vis_frame, pt1, pt2, (255, 255, 0), 2)
    
    return vis_frame


def multi_gpu_video_inference(video_path, output_path, detection_model_path, pose_config, pose_checkpoint,
                             detection_threshold=0.5, pose_threshold=0.3, num_gpus=3):
    """Multi-GPU video inference"""
    
    print(f"🚀 Starting Multi-GPU Video Inference")
    print(f"   GPUs: {num_gpus}")
    print(f"   Video: {video_path}")
    print(f"   Output: {output_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create queues
    frame_queue = Queue(maxsize=num_gpus * 4)  # Buffer frames
    result_queue = Queue()
    
    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        worker = Process(target=worker_process, args=(
            gpu_id, frame_queue, result_queue, detection_model_path,
            pose_config, pose_checkpoint, detection_threshold, pose_threshold
        ))
        worker.start()
        workers.append(worker)
    
    print(f"✅ Started {num_gpus} worker processes")
    
    # Read and queue frames
    frame_idx = 0
    frames_queued = 0
    
    # Queue initial frames
    while frames_queued < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_queue.put((frame_idx, frame))
        frame_idx += 1
        frames_queued += 1
        
        # Don't queue too many frames at once
        if frame_queue.qsize() >= num_gpus * 2:
            break
    
    cap.release()
    
    # Collect results and write video
    results_collected = 0
    frame_buffer = {}  # To maintain frame order
    next_frame_to_write = 0
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while results_collected < frames_queued:
        try:
            frame_idx, processed_frame = result_queue.get(timeout=10)
            frame_buffer[frame_idx] = processed_frame
            results_collected += 1
            
            # Write frames in order
            while next_frame_to_write in frame_buffer:
                out.write(frame_buffer[next_frame_to_write])
                del frame_buffer[next_frame_to_write]
                next_frame_to_write += 1
                pbar.update(1)
            
            # Queue more frames if available
            if frames_queued < total_frames:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames_queued)
                ret, frame = cap.read()
                if ret:
                    frame_queue.put((frames_queued, frame))
                    frames_queued += 1
                cap.release()
                
        except Exception as e:
            print(f"Error collecting results: {e}")
            break
    
    pbar.close()
    
    # Send poison pills to workers
    for _ in range(num_gpus):
        frame_queue.put(None)
    
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    
    out.release()
    
    print(f"✅ Multi-GPU video processing completed!")
    print(f"💾 Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Video Inference')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--output', default='output_multigpu.mp4', help='Output video path')
    parser.add_argument('--detection-model', 
                       default='mmpose/work_dirs/detection_training_extend/final_model.pth',
                       help='Detection model path')
    parser.add_argument('--pose-config',
                       default='mmpose/configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-11keypoints-expanded.py',
                       help='Pose config path')
    parser.add_argument('--pose-checkpoint',
                       default='mmpose/work_dirs/hrnet_silkworm_11keypoints_expanded/best_coco_AP_epoch_200.pth',
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
    
    multi_gpu_video_inference(
        args.video, args.output, args.detection_model,
        args.pose_config, args.pose_checkpoint,
        args.detection_threshold, args.pose_threshold, args.num_gpus
    )


if __name__ == '__main__':
    main()
