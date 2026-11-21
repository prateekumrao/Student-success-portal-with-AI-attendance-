import os
import sys
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

BASE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"dataset", "encodings", "notes", "static", "templates", "__pycache__"}


def find_repo():
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in SKIP]

        if "src" in dirs and "resources" in dirs:
            return root

    raise RuntimeError("Silent-Face-Anti-Spoofing folder (with src and resources) not found inside " + BASE)


REPO = find_repo()
sys.path.insert(0, REPO)

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.generate_patches import CropImage
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE
}


class LivenessChecker:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cropper = CropImage()
        self.models = []

        model_dir = os.path.join(REPO, "resources", "anti_spoof_models")

        for name in sorted(os.listdir(model_dir)):
            if not name.endswith(".pth"):
                continue

            h_input, w_input, model_type, scale = parse_model_name(name)
            model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(h_input, w_input)).to(self.device)

            state = torch.load(os.path.join(model_dir, name), map_location=self.device)

            if next(iter(state)).startswith("module."):
                state = OrderedDict((k[7:], v) for k, v in state.items())

            model.load_state_dict(state)
            model.eval()

            self.models.append((model, h_input, w_input, scale))

        if not self.models:
            raise RuntimeError("No anti-spoof models found in " + model_dir)

        detector_dir = os.path.join(REPO, "resources", "detection_model")
        prototxt = os.path.join(detector_dir, "deploy.prototxt")
        caffemodel = os.path.join(detector_dir, "Widerface-RetinaFace.caffemodel")

        self.detector = None

        try:
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        except Exception:
            try:
                self.detector = cv2.dnn.readNet(caffemodel, prototxt)
            except Exception as e:
                print("Warning: OpenCV face detector unavailable, using fallback box:", e)

    def detect(self, frame_bgr):
        if self.detector is None:
            return []

        height, width = frame_bgr.shape[:2]
        ratio = width / height
        img = frame_bgr

        if width * height >= 192 * 192:
            img = cv2.resize(frame_bgr, (int(192 * math.sqrt(ratio)), int(192 / math.sqrt(ratio))))

        self.detector.setInput(cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123)), "data")
        out = self.detector.forward("detection_out").reshape(-1, 7)

        boxes = []

        for row in out:
            if row[2] < 0.6:
                continue

            left, top = row[3] * width, row[4] * height
            right, bottom = row[5] * width, row[6] * height
            boxes.append([int(left), int(top), int(right - left + 1), int(bottom - top + 1)])

        return boxes

    def refine(self, bbox, detections):
        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        size = max(bbox[2], bbox[3])

        for d in detections or []:
            dx, dy = d[0] + d[2] / 2, d[1] + d[3] / 2

            if abs(dx - cx) <= size * 0.6 and abs(dy - cy) <= size * 0.6:
                return d

        x, y, w, h = bbox

        return [int(x - 0.1 * w), int(y - 0.25 * h), int(w * 1.2), int(h * 1.3)]

    def _run(self, frame_bgr, bbox):
        outputs = []

        for model, h_input, w_input, scale in self.models:
            patch = self.cropper.crop(
                org_img=frame_bgr,
                bbox=bbox,
                scale=scale,
                out_w=w_input,
                out_h=h_input,
                crop=scale is not None
            )

            tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs.append(F.softmax(model(tensor), dim=1).cpu().numpy()[0])

        return outputs

    def real_score(self, frame_bgr, bbox, detections=None):
        bbox = self.refine(bbox, detections)
        outputs = self._run(frame_bgr, bbox)

        return float(sum(o[1] for o in outputs) / len(outputs))

    def model_scores(self, frame_bgr, bbox, detections=None):
        bbox = self.refine(bbox, detections)

        return bbox, [[float(x) for x in o] for o in self._run(frame_bgr, bbox)]