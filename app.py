#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import logging
from scipy.signal import resample

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s'
                    )

# ============================================================
# 1) MODEL CONFIG
# ============================================================
MODEL_PATH = "model/task_1_activity_model_11_class.h5"
SOCIAL_MODEL_PATH = "model/social_signal_cnn_lstm.h5"

ACTIVITY_CLASSES = [
    "sittingStanding",
    "lyingLeft",
    "lyingRight",
    "lyingBack",
    "lyingStomach",
    "normalWalking",
    "running",
    "ascending",
    "descending",
    "shuffleWalking",
    "miscMovement"
]

SOCIAL_CLASSES = ["breathingNormally", "coughing", "hyperventilation", "other"]

ACTIVITY_MAP = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
STATIC_ACTIVITIES = {"sittingStanding", "lyingLeft", "lyingRight", "lyingBack", "lyingStomach"}

INPUT_WINDOW = 50
INPUT_DIM = 3
TRAIN_SAMPLING_RATE = 12.5  # RESpeck fixed freq
REALTIME_SAMPLING_RATE = None  # auto detect

# ============================================================
# 2) LOAD MODELS ONCE
# ============================================================
_activity_model = None
_social_model = None


def load_models():
    global _activity_model, _social_model
    try:
        _activity_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logging.info(f"[Model] Loaded activity model: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Failed to load activity model: {e}")
        raise

    try:
        _social_model = tf.keras.models.load_model(SOCIAL_MODEL_PATH, compile=False)
        logging.info(f"[Model] Loaded social model: {SOCIAL_MODEL_PATH}")
    except Exception as e:
        logging.error(f"Failed to load social model: {e}")
        _social_model = None


# ============================================================
# 3) PARSE INPUT DATA
# ============================================================
def parse_input(payload):
    """
    Accept any of:
      {"acc": [[x,y,z], ...]}
      {"x": [...], "y":[...], "z":[...]}
      {"samples": [{"x":...,"y":...,"z":...}]}
    """
    if "acc" in payload:
        return payload["acc"]

    if all(k in payload for k in ("x", "y", "z")):
        xs, ys, zs = payload["x"], payload["y"], payload["z"]
        return [[xs[i], ys[i], zs[i]] for i in range(len(xs))]

    if "samples" in payload:
        return [[s["x"], s["y"], s["z"]] for s in payload["samples"]]

    return []


# ============================================================
# 4) SAMPLING FIX (核心：保证前端信号 == 训练信号)
# ============================================================
def fix_sampling_rate(arr_raw):
    """
    前端 → 任意采样率（50Hz / 100Hz / 不规则）
    训练 → 固定 12.5Hz

    我们自动检测前端采样数，然后重采样到 50 samples 对应 4 秒。
    """
    N = arr_raw.shape[0]

    # 自动推测前端采样率（用4秒的action window来估）
    # 如果前端传 EXACTLY 50 samples → assume already 12.5 Hz
    # 如果传 200 samples → assume ~50Hz → 重采样
    if N <= 2:
        return np.zeros((INPUT_WINDOW, INPUT_DIM))

    global REALTIME_SAMPLING_RATE

    if REALTIME_SAMPLING_RATE is None:
        # 假设前端的窗口代表约4秒
        REALTIME_SAMPLING_RATE = N / 4.0
        logging.info(f"[Sampling] Auto-detected realtime sampling rate ≈ {REALTIME_SAMPLING_RATE:.2f} Hz")

    # 重采样到 50 点
    arr_fixed = resample(arr_raw, INPUT_WINDOW)
    return arr_fixed.astype("float32")


# ============================================================
# 5) WINDOW PREPROCESSING
# ============================================================
def prepare_window_activity(acc):
    acc = np.array(acc, dtype="float32")

    # ---- 重采样到固定50样本 ----
    arr = fix_sampling_rate(acc)

    # ---- reshape 成模型需要的 1×50×3 ----
    return arr.reshape(1, INPUT_WINDOW, INPUT_DIM)


def prepare_window_social(acc):
    acc = np.array(acc, dtype="float32")
    arr = fix_sampling_rate(acc)
    return arr.reshape(1, INPUT_WINDOW, INPUT_DIM)


# ============================================================
# 6) INFERENCE
# ============================================================
def predict_activity(arr):
    probs = _activity_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return ACTIVITY_CLASSES[idx], float(probs[idx]), idx


def predict_social(arr):
    if _social_model is None:
        return "Unavailable", 0.0, -1
    probs = _social_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return SOCIAL_CLASSES[idx], float(probs[idx]), idx


# ============================================================
# 7) FLASK API
# ============================================================
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "activity_model": MODEL_PATH,
        "social_model": SOCIAL_MODEL_PATH,
        "labels": ACTIVITY_CLASSES,
        "social_labels": SOCIAL_CLASSES,
        "sampling_rate_detected": REALTIME_SAMPLING_RATE
    })


@app.route("/classify", methods=["POST"])
def classify():
    t0 = time.time()
    payload = request.get_json(force=True)

    acc = parse_input(payload)
    if not acc or len(acc[0]) != 3:
        return jsonify({"ok": False, "error": "Input must contain accelerometer data [N,3]"}), 400

    arrA = prepare_window_activity(acc)
    labelA, confA, idxA = predict_activity(arrA)

    result = {
        "ok": True,
        "activity": labelA,
        "confidence": round(confA, 4),
        "class_index": idxA,
        "window_used": INPUT_WINDOW,
        "latency_ms": int((time.time() - t0) * 1000),
        "labels": ACTIVITY_CLASSES
    }

    # ---- static → detect social signal ----
    if labelA in STATIC_ACTIVITIES:
        arrS = prepare_window_social(acc)
        labelS, confS, idxS = predict_social(arrS)
        result["social_signal"] = {
            "label": labelS,
            "confidence": round(confS, 4),
            "class_index": idxS,
            "labels": SOCIAL_CLASSES
        }

    return jsonify(result), 200


# ============================================================
# 8) START
# ============================================================
if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
