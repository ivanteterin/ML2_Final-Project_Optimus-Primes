import os
import shutil
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

# --- Configuration and Model Architecture (copied from notebook) ---

# Set random seeds for reproducibility (important for model loading consistency)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Global constants from the notebook context
POSE_MODEL = 'yolo11n-pose.pt'
CONF_THRESHOLD = 0.15
MAX_LEN = 103 # From notebook context `MAX_LEN`

# Feature configuration from the best performing experiment in the notebook
# (best_cfg = {'exp_name': 'transformer_skill_only', 'multitask': False, 'use_vel_acc': True, 'use_conf': True, 'use_normalization': True})
FEATURE_CONFIG = {
    'use_vel_acc': True,
    'use_conf': True,
    'use_normalization': True
}

# Set device to CPU for general deployment, or 'cuda' if GPU is available
DEVICE = 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerSkillMultiTask(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.2, num_actions=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.skill_head = nn.Linear(d_model, 2)
        self.action_head = nn.Linear(d_model, num_actions) # num_actions=4 for the 4 classes

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = self.norm(z)
        pooled = z.mean(dim=1)
        skill_logits = self.skill_head(pooled)
        action_logits = self.action_head(pooled)
        return skill_logits, action_logits

# --- Preprocessing Functions (copied from notebook) ---

def select_best_person(kps_xy_all, kps_conf_all):
    # Select the person with the highest average confidence score across keypoints
    return int(np.argmax(kps_conf_all.mean(axis=1)))

def extract_keypoints_for_video(model_det, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model_det(frame, verbose=False)[0]
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            xy_all = r.keypoints.xy.cpu().numpy()
            if r.keypoints.conf is not None:
                conf_all = r.keypoints.conf.cpu().numpy()
            else:
                conf_all = np.ones((xy_all.shape[0], xy_all.shape[1]), dtype=np.float32)

            pidx = select_best_person(xy_all, conf_all)
            xy = xy_all[pidx].astype(np.float32)
            conf = conf_all[pidx].astype(np.float32).reshape(-1, 1)

            low = conf[:, 0] < CONF_THRESHOLD
            xy[low] = 0.0
            conf[low] = 0.0

            arr = np.concatenate([xy, conf], axis=1)
        else:
            # No keypoints detected, return zeros for 17 keypoints and 3 dimensions (x, y, conf)
            arr = np.zeros((17, 3), dtype=np.float32)

        frames.append(arr)
        idx += 1

    cap.release()
    if not frames:
        return None # No frames processed or video was empty
    return np.stack(frames)

def extract_features(arr, use_vel_acc=True, use_conf=True, use_normalization=True):
    coords = arr[..., :2].astype(np.float32)
    conf = arr[..., 2].astype(np.float32)

    # body-centered normalization (can be disabled for ablation)
    if use_normalization:
        lsh = coords[:,5,:]
        rsh = coords[:,6,:]
        lhip = coords[:,11,:]
        rhip = coords[:,12,:]
        center = (lsh + rsh + lhip + rhip) / 4.0
        scale = np.linalg.norm(lsh - rhip, axis=1, keepdims=True) + 1e-6
        norm = (coords - center[:,None,:]) / scale[:,None,:]
    else:
        norm = coords  # raw pixel coords, no normalization

    parts = [norm.reshape(norm.shape[0], -1)]

    if use_vel_acc:
        vel = np.diff(norm, axis=0)
        vel = np.concatenate([vel, vel[-1:]], axis=0) if len(vel) else np.zeros_like(norm)
        acc = np.diff(vel, axis=0)
        acc = np.concatenate([acc, acc[-1:]], axis=0) if len(acc) else np.zeros_like(norm)
        parts.append(vel.reshape(vel.shape[0], -1))
        parts.append(acc.reshape(acc.shape[0], -1))

    if use_conf:
        parts.append(conf)

    feat = np.concatenate(parts, axis=1).astype(np.float32)
    return feat

def pad_or_truncate(seq, max_len):
    t, f = seq.shape
    if t >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - t, f), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)

# --- FastAPI Application ---

app = FastAPI()

# Load models globally to avoid re-loading on each request
pose_detector = YOLO(POSE_MODEL)

# Determine input_dim for Transformer model
# This assumes the feature extraction without any actual data for now to get dimensions
# A more robust way would be to save input_dim during training or calculate from constants
# Assuming 17 keypoints * 2 (x,y) + 17 (conf) + 17*2 (vel) + 17*2 (acc) = 17*6 + 17 = 119
# If use_vel_acc is False: 17*2 (x,y) + 17 (conf) = 51
# If use_conf is False: 17*2 (x,y) + 17*2 (vel) + 17*2 (acc) = 102

input_dim = 0
if FEATURE_CONFIG['use_normalization']:
    input_dim += 17 * 2 # Normalized x, y coordinates
else:
    input_dim += 17 * 2 # Raw x, y coordinates

if FEATURE_CONFIG['use_vel_acc']:
    input_dim += 17 * 2 # Velocity
    input_dim += 17 * 2 # Acceleration

if FEATURE_CONFIG['use_conf']:
    input_dim += 17 # Confidence scores

# Initialize Transformer model with the determined input_dim
model_exportable = TransformerSkillMultiTask(input_dim=input_dim, num_actions=4).to(DEVICE)

# Load the pre-trained state dictionary
MODEL_PATH = "model_exportable_state_dict.pth"
if not Path(MODEL_PATH).exists():
    # Attempt to download the YOLO pose model if not already present
    # YOLO constructor usually handles this, but explicitly downloading here for the .pt file.
    # For yolo11n-pose.pt, it's typically downloaded to ~/.config/Ultralytics/yolo11n-pose.pt
    # However, the model_exportable_state_dict.pth is YOUR trained model.
    # You need to ensure it's in the same directory as main.py or provide its path.
    print(f"Error: Model state dictionary not found at {MODEL_PATH}")
    # Exit or raise error, cannot proceed without the model
else:
    model_exportable.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model_exportable.eval()
    print(f"Model '{MODEL_PATH}' loaded successfully on {DEVICE}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Skill Prediction API! Upload a video to /predict"}

@app.post("/predict", response_model=Dict[str, Any])
async def predict_skill(video_file: UploadFile = File(...)):
    temp_video_path = Path(f"./temp_{video_file.filename}")
    
    # Save the uploaded video temporarily
    try:
        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # 1. Extract Keypoints
        keypoints_array = extract_keypoints_for_video(pose_detector, temp_video_path)

        if keypoints_array is None or keypoints_array.shape[0] == 0:
            return {"error": "No keypoints extracted from video or video is invalid."}

        # 2. Extract Features
        features = extract_features(
            keypoints_array,
            use_vel_acc=FEATURE_CONFIG['use_vel_acc'],
            use_conf=FEATURE_CONFIG['use_conf'],
            use_normalization=FEATURE_CONFIG['use_normalization']
        )

        # 3. Pad/Truncate
        padded_features = pad_or_truncate(features, MAX_LEN)

        # 4. Convert to Tensor and infer
        X_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dimension

        with torch.no_grad():
            skill_logits, _ = model_exportable(X_tensor)
            skill_probs = torch.softmax(skill_logits, dim=1)
            pred_skill_idx = skill_probs.argmax(dim=1).item()
            prob_expert = skill_probs[0, 1].item() # Probability of class 1 (expert)

        skill_label = 'expert' if pred_skill_idx == 1 else 'beginner'
        
        return {
            "filename": video_file.filename,
            "predicted_skill_label": skill_label,
            "expert_probability": round(prob_expert, 4)
        }
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
    finally:
        # Clean up the temporary video file
        if temp_video_path.exists():
            os.remove(temp_video_path)

if __name__ == '__main__':
    # To run this with uvicorn directly, you would use:
    # uvicorn main:app --host 0.0.0.0 --port 8000
    # For this demonstration, we just define the app.
    pass
