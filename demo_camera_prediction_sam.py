# realtime_yolo_sam_roi_fallback_gpu.py
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import pyrealsense2 as rs
from collections import deque

# ===================== 설정 =====================
YOLO_WEIGHTS = "best.pt"
TARGET_NAME  = "door-handle"

IMGSZ_FULL   = 512
IMGSZ_ROI    = 896
FULL_EVERY_N = 12
ROI_SCALE    = 2.0

CONF_ENTER   = 0.22
CONF_STAY    = 0.12
IOU_GATE     = 0.08

SAM_CKPT     = "sam_vit_h_4b8939.pth"
PAD_FOR_SAM  = 12
DRAW_ALPHA   = 0.35

USE_CUDA     = torch.cuda.is_available()
DEVICE_YOLO  = 0 if USE_CUDA else 'cpu'
USE_FP16     = USE_CUDA
torch.backends.cudnn.benchmark = True

USE_CV2_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

FIX_EXPOSURE = False
EXPOSURE_US  = 8000
GAIN         = 64

# ===================== 유틸 =====================
def clamp_box(box, w, h, pad=0):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return [x1, y1, x2, y2]

def expand_box(box, w, h, scale=2.0):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    nw, nh = bw*scale, bh*scale
    nx1, ny1 = int(round(cx - nw/2)), int(round(cy - nh/2))
    nx2, ny2 = int(round(cx + nw/2)), int(round(cy + nh/2))
    return clamp_box([nx1, ny1, nx2, ny2], w, h)

def iou(a, b):
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih  = max(0, ix2-ix1), max(0, iy2-iy1)
    inter   = iw * ih
    ua = max(0, ax2-ax1) * max(0, ay2-ay1)
    ub = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (ua + ub - inter + 1e-6)

def letterbox(image_bgr, new_size):
    h0, w0 = image_bgr.shape[:2]
    r = min(new_size/h0, new_size/w0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2; dh /= 2
    resized = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114,114,114))
    def back(xyxy):
        x1, y1, x2, y2 = xyxy
        x1 = (x1-left)/r; y1 = (y1-top)/r
        x2 = (x2-left)/r; y2 = (y2-top)/r
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return padded, back

def crop_with_map(frame_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    def back(xyxy):
        u1, v1, u2, v2 = xyxy
        return [x1+int(round(u1)), y1+int(round(v1)),
                x1+int(round(u2)), y1+int(round(v2))]
    return crop, back

def largest_component_mask(mask_bool):
    m = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num <= 1: return mask_bool
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == idx

def post_mask(mask_bool):
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    return largest_component_mask(m > 0)

def bgr_to_rgb(frame_bgr):
    if USE_CV2_CUDA:
        g = cv2.cuda_GpuMat(); g.upload(frame_bgr)
        rgb_g = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2RGB)
        return rgb_g.download()
    else:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def overlay_mask_bgr(frame_bgr, mask_bool, alpha=0.35, color=(0,255,0)):
    if not np.any(mask_bool):
        return frame_bgr
    if USE_CV2_CUDA:
        g_frame = cv2.cuda_GpuMat(); g_frame.upload(frame_bgr)
        m = (mask_bool.astype(np.uint8) * 255)
        g_mask = cv2.cuda_GpuMat(); g_mask.upload(m)
        color_img = np.zeros_like(frame_bgr, dtype=np.uint8); color_img[:] = color
        g_color = cv2.cuda_GpuMat(); g_color.upload(color_img)
        g_masked = cv2.cuda.bitwise_and(g_color, g_color, mask=g_mask)
        g_out = cv2.cuda.addWeighted(g_frame, 1.0, g_masked, alpha, 0)
        return g_out.download()
    else:
        overlay = frame_bgr.copy()
        color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_layer[mask_bool] = color
        return cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)

# ===================== 메인 =====================
def main():
    yolo = YOLO(YOLO_WEIGHTS)
    if USE_CUDA:
        yolo.to('cuda')
        yolo.model.float()

    name_map = yolo.model.names
    target_idx = None
    for k, v in name_map.items():
        if str(v).strip().lower() == TARGET_NAME:
            target_idx = int(k); break
    if target_idx is None:
        raise RuntimeError(f"'{TARGET_NAME}' 클래스가 YOLO 모델에 없음: {name_map}")

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    if USE_CUDA: sam.to('cuda')
    predictor = SamPredictor(sam)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    if FIX_EXPOSURE:
        sens = profile.get_device().query_sensors()[1]
        try:
            sens.set_option(rs.option.enable_auto_exposure, 0)
            sens.set_option(rs.option.exposure, EXPOSURE_US)
            sens.set_option(rs.option.gain, GAIN)
        except Exception:
            pass
    align_to_color = rs.align(rs.stream.color)

    prev_box = None
    have = False
    fps_buf = deque(maxlen=30)
    frame_idx = 0

    try:
        with torch.inference_mode():
            while True:
                t0 = time.time()
                frames = align_to_color.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_bgr = np.asanyarray(color_frame.get_data())
                H, W = frame_bgr.shape[:2]
                frame_rgb = bgr_to_rgb(frame_bgr)

                use_full = (not have) or (frame_idx % FULL_EVERY_N == 0)
                best_box, best_conf = None, -1.0

                if not use_full and prev_box is not None:
                    roi_box = expand_box(prev_box, W, H, scale=ROI_SCALE)
                    crop_bgr, back_map = crop_with_map(frame_bgr, roi_box)
                    lb, back = letterbox(crop_bgr, IMGSZ_ROI)
                    res = yolo.predict(
                        lb, imgsz=IMGSZ_ROI, classes=[target_idx],
                        conf=CONF_STAY, verbose=False, device=DEVICE_YOLO,
                        half=False, agnostic_nms=True
                    )[0]
                    if res.boxes is not None and res.boxes.xyxy is not None:
                        xyxy = res.boxes.xyxy.detach().cpu().numpy()
                        confs = res.boxes.conf.detach().cpu().numpy()
                        for b, cf in zip(xyxy, confs):
                            cand = back(b)
                            cand = back_map(cand)
                            if cf > best_conf and (prev_box is None or iou(prev_box, cand) >= IOU_GATE):
                                best_conf = float(cf); best_box = cand

                if use_full or best_box is None:
                    lb, back = letterbox(frame_bgr, IMGSZ_FULL)
                    res = yolo.predict(
                        lb, imgsz=IMGSZ_FULL, classes=[target_idx],
                        conf=CONF_ENTER if not have else CONF_STAY,
                        verbose=False, device=DEVICE_YOLO,
                        half=USE_FP16, agnostic_nms=True
                    )[0]
                    if res.boxes is not None and res.boxes.xyxy is not None:
                        xyxy = res.boxes.xyxy.detach().cpu().numpy()
                        confs = res.boxes.conf.detach().cpu().numpy()
                        for b, cf in zip(xyxy, confs):
                            cand = back(b)
                            if cf > best_conf:
                                best_conf = float(cf); best_box = cand

                accept = False
                if have:
                    if best_box is not None and best_conf >= CONF_STAY:
                        accept = True
                    else:
                        have = False; prev_box = None
                else:
                    if best_box is not None and best_conf >= CONF_ENTER:
                        have = True; accept = True

                overlay = frame_bgr.copy()

                if accept and best_box is not None:
                    x1, y1, x2, y2 = clamp_box(best_box, W, H, pad=PAD_FOR_SAM)
                    predictor.set_image(frame_rgb)
                    masks, _, _ = predictor.predict(
                        box=np.array([x1, y1, x2, y2])[None, :],
                        multimask_output=False
                    )
                    mask = post_mask(masks[0].astype(bool))
                    overlay = overlay_mask_bgr(overlay, mask, alpha=DRAW_ALPHA, color=(0,255,0))

                    ys, xs = np.where(mask)
                    if len(xs) > 0:
                        mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
                        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (255,0,0), 2)
                        prev_box = [mx1, my1, mx2, my2]
                    else:
                        prev_box = best_box

                    cv2.putText(overlay, f"{TARGET_NAME} conf={best_conf:.2f}", (12,24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
                else:
                    cv2.putText(overlay, f"No {TARGET_NAME}", (12,24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                dt = time.time() - t0
                fps_buf.append(1.0 / max(dt, 1e-6))
                fps = sum(fps_buf) / len(fps_buf)
                cv2.putText(overlay, f"FPS {fps:.1f}", (12,48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)

                cv2.imshow("ROI-first YOLO -> SAM (GPU-boost)", overlay)
                frame_idx += 1
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
