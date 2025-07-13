#!/usr/bin/env python3
# Copyright (c) OpenMMLab.  Multi-GPU fast demo.

import os
import argparse
import math
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp

import cv2
import numpy as np
from decord import VideoReader, cpu


def parse_args():
    p = argparse.ArgumentParser()
    # 必須 4 ファイル
    p.add_argument('det_config')
    p.add_argument('det_ckpt')
    p.add_argument('pose_config')
    p.add_argument('pose_ckpt')

    p.add_argument('--input', required=True, help='input video')
    p.add_argument('--output', default='output.mp4', help='output video')
    p.add_argument('--gpus', nargs='+', default=['0'],
                   help='GPU id list, e.g. 0 1 2')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--bbox-thr', type=float, default=0.3)
    p.add_argument('--kpt-thr',  type=float, default=0.3)
    return p.parse_args()


def worker(proc_idx: int, gpu_id: str, frame_range: Tuple[int, int],
           args, tmp_mp4: str):
    """1 プロセス＝1 GPU で区間を処理して tmp_mp4 に書く"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id       # 必ず先頭で

    # ----- heavy imports はここで行う -----
    import torch
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model as init_pose_model, inference_topdown
    from mmpose.structures import merge_data_samples
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.registry import VISUALIZERS
    from mmengine import track_iter_progress
    import mmcv

    torch.backends.cudnn.benchmark = True

    # -------- models ----------
    det = init_detector(args.det_config, args.det_ckpt, device='cuda:0')
    det.cfg = adapt_mmdet_pipeline(det.cfg)

    pose = init_pose_model(
        args.pose_config, args.pose_ckpt, device='cuda:0',
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    vis = VISUALIZERS.build(pose.cfg.visualizer)
    vis.set_dataset_meta(pose.dataset_meta)
    vis.radius, vis.line_width = 3, 2

    # -------- I/O -------------
    vr = VideoReader(args.input, ctx=cpu(0))
    fps = vr.get_avg_fps()
    h, w = vr[0].shape[0:2]

    vw = cv2.VideoWriter(
        tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    bsz = args.batch_size
    buf_frames, buf_bboxes = [], []

    for f_idx in track_iter_progress(range(*frame_range)):
        frame = vr[f_idx].asnumpy()[:, :, ::-1]          # RGB→BGR
        buf_frames.append(frame)

        det_res = inference_detector(det, frame)
        inst = det_res.pred_instances.cpu().numpy()
        b = np.concatenate((inst.bboxes, inst.scores[:, None]), 1)
        m = np.logical_and(inst.labels == 0, inst.scores > args.bbox_thr)
        buf_bboxes.append(b[m][:, :4])

        last = (f_idx == frame_range[1] - 1)
        if len(buf_frames) == bsz or last:
            pose_res = inference_topdown(
                pose, buf_frames, buf_bboxes,
                batch_size=bsz, return_vis=False)

            for f, r in zip(buf_frames, pose_res):
                ds = merge_data_samples(r)
                vis.add_datasample(
                    'vis', mmcv.bgr2rgb(f), data_sample=ds,
                    draw_gt=False, kpt_thr=args.kpt_thr, show=False)
                out = vis.get_image()
                vw.write(mmcv.rgb2bgr(out))
            buf_frames.clear()
            buf_bboxes.clear()

    vw.release()
    print(f'[GPU {gpu_id}] finished {frame_range}')


def split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
    size = math.ceil(total / parts)
    return [(i * size, min((i + 1) * size, total)) for i in range(parts)]


def concat_videos(tmp_files: List[str], out_path: str):
    list_file = tempfile.mktemp(suffix='.txt')
    with open(list_file, 'w') as f:
        for p in tmp_files:
            f.write(f"file '{p}'\n")
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
           '-i', list_file, '-c', 'copy', out_path]
    subprocess.run(cmd, check=True)
    os.remove(list_file)


def main():
    args = parse_args()
    gpus = args.gpus
    num_gpu = len(gpus)

    # 動画基本情報を取得
    vr = VideoReader(args.input, ctx=cpu(0))
    total_frames = len(vr)

    ranges = split_ranges(total_frames, num_gpu)

    tmp_files = [tempfile.mktemp(suffix=f'_{i}.mp4') for i in range(num_gpu)]
    procs = []
    for i, (gpu, fr) in enumerate(zip(gpus, ranges)):
        p = mp.Process(target=worker,
                       args=(i, gpu, fr, args, tmp_files[i]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 連結して完成
    concat_videos(tmp_files, args.output)
    for p in tmp_files:
        os.remove(p)
    print(f'>>> DONE: {args.output}')


if __name__ == '__main__':
    main()