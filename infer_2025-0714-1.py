from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer
import cv2, numpy as np, os

cfg  = 'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py'
ckpt = 'work_dirs/hrnet_silkworm11/epoch_210.pth'
img  = 'test1.jpg'          # 推論したい画像

model = init_model(cfg, ckpt, device='cuda:0')  # or 'cpu'

# Load image
img_array = cv2.imread(img)
h, w = img_array.shape[:2]
person_results = [np.array([0, 0, w, h], dtype=np.float32)]   # Full image bbox

# Run inference
preds = inference_topdown(model, img, person_results)

# Initialize visualizer
visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(model.dataset_meta)

# Visualize results
out = 'test1_pose.jpg'
visualizer.add_datasample(
    'result',
    img_array,
    data_sample=preds[0],
    draw_gt=False,
    draw_heatmap=False,
    show_kpt_idx=True,
    skeleton_style='mmpose'
)

# Get the visualization result and save it
vis_result = visualizer.get_image()
cv2.imwrite(out, vis_result)
print('saved to', os.path.abspath(out))