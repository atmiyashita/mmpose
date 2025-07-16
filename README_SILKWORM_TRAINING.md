# Silkworm Detection and Pose Estimation Training Guide

This guide explains how to train and test both bounding box detection and pose estimation models for silkworm analysis.

## 📁 Directory Structure

```
mmpose/
├── configs/silkworm/                    # Configuration files
├── work_dirs/                           # Training outputs
│   ├── detection_training_extend/       # Detection model outputs
│   └── hrnet_silkworm_4keypoints/      # Pose model outputs
├── data/silkworm/                       # Dataset (outside mmpose/)
│   ├── detection_data/                  # Detection annotations
│   ├── keypoints_merged/               # 11-keypoint dataset
│   └── keypoints_4point/               # 4-keypoint dataset
└── models/detection/                    # Pretrained models
```

## 🎯 Bounding Box Detection Training

### Prerequisites
- Detection annotations in COCO format
- Images in a directory structure

### 1. Prepare Detection Dataset
```bash
# Check your detection annotations
python check_label_consistency.py --strategy extend

# This creates: work_dirs/detection_data/instances_consistent.json
```

### 2. Train Detection Model
```bash
# Quick training with default settings
python start_detection_training.py --strategy extend --epochs 30

# Or manual training with custom parameters
python train_detection.py \
    --annotation-file work_dirs/detection_data/instances_consistent.json \
    --num-classes 5 \
    --epochs 30 \
    --batch-size 4 \
    --lr 0.001 \
    --output-dir work_dirs/detection_training_extend
```

### 3. Test Detection Model
```bash
# Test on single image
python test_detection_pose.py test1.jpg \
    --detection-model work_dirs/detection_training_extend/final_model.pth \
    --output detection_result.jpg

# Compare old vs new detection models
python compare_detection_models.py
```

### Detection Model Outputs
- `work_dirs/detection_training_extend/final_model.pth` - Final trained model
- `work_dirs/detection_training_extend/model_epoch_X.pth` - Epoch checkpoints
- `work_dirs/detection_training_extend/training_curves.png` - Training plots

## 🎯 Pose Estimation Training

### Prerequisites
- Keypoint annotations in COCO format
- Corresponding images

### 1. Prepare Keypoint Dataset

#### Option A: Use 11-keypoint dataset
```bash
# Merge multiple annotation parts
python prepare_keypoint_training.py \
    --part1-dir ../data/silkworm/keypoints_traindata/part1 \
    --part2-dir ../data/silkworm/keypoints_traindata/part2 \
    --output-dir ../data/silkworm/keypoints_merged
```

#### Option B: Convert to 4-keypoint dataset (Recommended)
```bash
# Convert 11-keypoint to 4-keypoint (H1, T1, A2, A9)
python convert_to_4keypoints.py \
    --input-dir ../data/silkworm/keypoints_merged \
    --output-dir ../data/silkworm/keypoints_4point
```

### 2. Train Pose Models

#### Train 4-keypoint model (Recommended)
```bash
python tools/train.py \
    configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py \
    --work-dir work_dirs/hrnet_silkworm_4keypoints
```

#### Train 11-keypoint model
```bash
python tools/train.py \
    configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-keypoints-improved.py \
    --work-dir work_dirs/hrnet_silkworm_keypoints_improved
```

### 3. Test Pose Models
```bash
# Test 4-keypoint vs 11-keypoint comparison
python test_4keypoint_model.py test1.jpg

# Test complete detection + pose pipeline
python test_detection_pose.py test1.jpg --output pose_result.jpg
```

### Pose Model Outputs
- `work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_X.pth` - Best model
- `work_dirs/hrnet_silkworm_4keypoints/epoch_X.pth` - Final model
- Training logs and metrics in work_dirs

## 🎬 Video Processing

### Process video with trained models
```bash
# Use default trained models
python infer_detection_pose.py input_video.mp4 --output result_video.mp4

# Specify custom models
python infer_detection_pose.py input_video.mp4 \
    --detection-model work_dirs/detection_training_extend/final_model.pth \
    --pose-checkpoint work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_130.pth \
    --output result_video.mp4
```

## 📊 Model Evaluation

### Compare model performance
```bash
# Detection model comparison
python simple_model_comparison.py test1.jpg

# Pose model comparison  
python test_4keypoint_model.py test1.jpg

# Complete system test
python test_model_improvements.py test1.jpg
```

## 🔧 Configuration Files

### Detection Training Config
- `configs/silkworm/detection_config.py` - Detection training parameters

### Pose Training Configs
- `configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py` - 4-keypoint model
- `configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-keypoints-improved.py` - 11-keypoint model

### Dataset Configs
- `configs/_base_/datasets/silkworm4_coco.py` - 4-keypoint dataset definition
- `configs/_base_/datasets/silkworm11_coco.py` - 11-keypoint dataset definition

## 🎯 Recommended Workflow

### 1. Detection Training
```bash
cd mmpose
python start_detection_training.py --strategy extend --epochs 30
```

### 2. Pose Training (4-keypoint)
```bash
# Prepare dataset
python convert_to_4keypoints.py

# Train model
python tools/train.py configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py \
    --work-dir work_dirs/hrnet_silkworm_4keypoints
```

### 3. Test Complete System
```bash
# Test on image
python test_detection_pose.py test1.jpg --output final_result.jpg

# Test on video
python infer_detection_pose.py video.mp4 --output final_video.mp4
```

## 📈 Expected Performance

### Detection Model
- **Training Loss**: ~0.02 (excellent convergence)
- **Confidence**: >0.7 for good detections
- **Classes**: 5 (background, kaiko_live, kaiko_pao1, kaiko_rhi, kaiko)

### 4-Keypoint Pose Model
- **COCO AP**: ~0.24 (good for specialized dataset)
- **Reliability**: 75% (3/4 keypoints with >0.3 confidence)
- **Keypoints**: H1 (head), T1 (thorax), A2 (abdomen), A9 (tail)

## 🚀 Quick Start Commands

```bash
# Complete training pipeline
cd mmpose

# 1. Train detection
python start_detection_training.py --strategy extend --epochs 30

# 2. Prepare 4-keypoint dataset
python convert_to_4keypoints.py

# 3. Train 4-keypoint pose
python tools/train.py configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py \
    --work-dir work_dirs/hrnet_silkworm_4keypoints

# 4. Test complete system
python test_detection_pose.py test1.jpg --output result.jpg
```

## 📝 Notes

- **4-keypoint model is recommended** over 11-keypoint for better reliability
- **Detection model uses "extend" strategy** to maintain compatibility with pretrained classes
- **All scripts use the best trained models by default**
- **GPU training is recommended** for faster convergence
- **Validation is performed every 10 epochs** during training
