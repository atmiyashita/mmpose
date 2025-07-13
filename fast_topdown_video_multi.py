#!/usr/bin/env python3
# fast_topdown_video_multi.py  –  multi-GPU video demo (robust version)

import os, math, shutil, tempfile, argparse, subprocess, multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import cv2
from decord import VideoReader, cpu

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('det_config');    p.add_argument('det_ckpt')
    p.add_argument('pose_config');   p.add_argument('pose_ckpt')
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='output.mp4')
    p.add_argument('--gpus', nargs='+', default=['0'])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--bbox-thr', type=float, default=0.3)
    p.add_argument('--kpt-thr',  type=float, default=0.3)
    return p.parse_args()

# ------------------------------------------------------------
# Worker (=1GPU)
# ------------------------------------------------------------
def worker(idx: int, gpu: str, frange: Tuple[int, int], args, tmp_mp4: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu          # lock GPU
    try:
        # --- heavy imports（各プロセスで遅延読み込み） ---
        import torch, mmcv, numpy as np
        from mmengine import track_iter_progress
        from mmdet.apis import init_detector, inference_detector
        from mmpose.apis import init_model as init_pose_model, inference_topdown
        from mmpose.structures import merge_data_samples
        from mmpose.utils import adapt_mmdet_pipeline
        from mmpose.registry import VISUALIZERS
        torch.backends.cudnn.benchmark = True
        device = 'cuda:0'

        # --- models ---
        det  = init_detector(args.det_config,  args.det_ckpt,  device=device)
        det.cfg = adapt_mmdet_pipeline(det.cfg)
        pose = init_pose_model(
            args.pose_config, args.pose_ckpt, device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
        vis = VISUALIZERS.build(pose.cfg.visualizer)
        vis.set_dataset_meta(pose.dataset_meta);  vis.radius = 3; vis.line_width = 2

        # --- video I/O ---
        vr  = VideoReader(args.input, ctx=cpu(0))
        fps = vr.get_avg_fps();  h, w = vr[0].shape[:2]
        vw  = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        assert vw.isOpened(), f'VideoWriter failed on {tmp_mp4}'

        buf_f, buf_b = [], [];  bsz = args.batch_size

        # --- main loop ---
        for fi in track_iter_progress(range(*frange), description=f'GPU{gpu}'):
            f = vr[fi].asnumpy()[:, :, ::-1];  buf_f.append(f)         # RGB→BGR
            dres = inference_detector(det, f).pred_instances.cpu().numpy()
            box  = np.concatenate([dres.bboxes, dres.scores[:, None]], 1)
            keep = np.logical_and(dres.labels == 0, dres.scores > args.bbox_thr)
            buf_b.append(box[keep][:, :4])

            flush = len(buf_f) == bsz or fi == frange[1] - 1
            if flush:
                pres = inference_topdown(pose, buf_f, buf_b,
                                         batch_size=bsz, return_vis=False)
                for frm, rs in zip(buf_f, pres):
                    vis.add_datasample('vis', mmcv.bgr2rgb(frm),
                        data_sample=merge_data_samples(rs),
                        draw_gt=False, kpt_thr=args.kpt_thr, show=False)
                    vw.write(mmcv.rgb2bgr(vis.get_image()))
                buf_f.clear(); buf_b.clear()

        vw.release()
        if not Path(tmp_mp4).exists() or Path(tmp_mp4).stat().st_size == 0:
            raise RuntimeError(f'Worker {gpu}: tmp video not written')
        print(f'[GPU {gpu}] completed frames {frange}')

    except Exception:
        import traceback, sys
        traceback.print_exc();  sys.exit(1)

# ------------------------------------------------------------
# util
# ------------------------------------------------------------
def split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
    step = math.ceil(total / parts)
    return [(i*step, min((i+1)*step, total)) for i in range(parts)]

def concat_videos(tmp: List[str], dst: str):
    txt = tempfile.mktemp(suffix='.txt')
    with open(txt, 'w') as f:
        for p in tmp:
            f.write(f"file '{p}'\n")
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', txt, '-c', 'copy', dst], check=True)
    os.remove(txt)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args();  n_gpu = len(args.gpus)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total = len(VideoReader(args.input, ctx=cpu(0)))
    ranges = split_ranges(total, n_gpu)
    tmp_files = [tempfile.mktemp(suffix=f'_{i}.mp4') for i in range(n_gpu)]

    procs = []
    for i, (gid, fr) in enumerate(zip(args.gpus, ranges)):
        p = mp.Process(target=worker, args=(i, gid, fr, args, tmp_files[i]))
        p.start();  procs.append(p)
    for p in procs: p.join()

    # 連結 or 単体コピー
    if n_gpu == 1:
        shutil.move(tmp_files[0], args.output)     # cross-device OK
    else:
        concat_videos(tmp_files, args.output)
        for p in tmp_files: Path(p).unlink()
    print('>>> DONE:', args.output)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # PyTorch 推奨
    main()