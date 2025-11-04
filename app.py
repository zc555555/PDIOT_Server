# import os, time, json, hashlib, pathlib, tempfile
# from typing import List, Dict, Any, Tuple
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import logging
#
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s: %(message)s'
# )
#
# MODEL_PATH = os.path.join("model", "task_1_activity_model_11_class.h5")
# MODEL_URL    = ""
# INPUT_WINDOW = int(os.getenv("INPUT_WINDOW", "25"))
# INPUT_DIM    = int(os.getenv("INPUT_DIM", "3"))
# # LABELS = os.getenv(
# #     "LABELS",
# #     "ascending,descending,lyingBack,lyingLeft,lyingRight,lyingStomach,miscMovement,normalWalking,running,shuffleWalking,sittingStanding"
# # ).split(",")
#
# LABELS = [
#     "sittingStanding",
#     "lyingLeft",
#     "lyingRight",
#     "lyingBack",
#     "lyingStomach",
#     "normalWalking",
#     "running",
#     "ascending",
#     "descending",
#     "shuffleWalking",
#     "miscMovement"
# ]
#
#
# THRESH_PAD   = int(os.getenv("THRESH_PAD", "0"))
# NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "false"
#
#
# app = Flask(__name__)
# CORS(app)
#
#
# def _download_to(path: str, url: str) -> str:
#     import requests
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     r = requests.get(url, timeout=60)
#     r.raise_for_status()
#     with open(path, "wb") as f:
#         f.write(r.content)
#     return path
#
#
# _model = None
# _loaded_from = None
# def load_model_once() -> Tuple[str, str]:
#     """
#     return (status, source)
#     status: 'ok'|'missing'|'error'
#     source: 'local'|'url'|'none'
#     """
#     global _model, _loaded_from
#     if _model is not None:
#         return "ok", _loaded_from
#
#     model_path = MODEL_PATH
#
#     if (not os.path.exists(model_path)) and MODEL_URL:
#         try:
#             os.makedirs(os.path.dirname(model_path), exist_ok=True)
#             _download_to(model_path, MODEL_URL)
#             _loaded_from = "url"
#         except Exception as e:
#             app.logger.exception(f"Download model failed: {e}")
#             return "error", "url"
#     else:
#         _loaded_from = "local" if os.path.exists(model_path) else "none"
#
#     if not os.path.exists(model_path):
#         return "missing", _loaded_from
#
#     try:
#
#         from tensorflow.keras.models import load_model
#
#         _model = load_model(model_path, compile=False)
#
#         return "ok", _loaded_from
#     except Exception as e:
#         app.logger.exception(f"Load model failed: {e}")
#         return "error", _loaded_from
#
#
# def parse_acc_data(payload: Dict[str, Any]) -> List[List[float]]:
#     # A) {"acc": [[x,y,z], ...]}
#     if isinstance(payload, dict) and "acc" in payload and isinstance(payload["acc"], list):
#         return payload["acc"]
#     # B) {"x":[...],"y":[...],"z":[...]}
#     if isinstance(payload, dict) and all(k in payload for k in ("x","y","z")):
#         xs, ys, zs = payload["x"], payload["y"], payload["z"]
#         if len(xs) == len(ys) == len(zs) and len(xs) > 0:
#             return [[float(xs[i]), float(ys[i]), float(zs[i])] for i in range(len(xs))]
#     # C) {"samples":[{"time":t,"x":...,"y":...,"z":...}, ...]}
#     if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
#         out = []
#         for s in payload["samples"]:
#             if all(k in s for k in ("x","y","z")):
#                 out.append([float(s["x"]), float(s["y"]), float(s["z"])])
#         return out
#     return []
#
#
# def prepare_window(acc_xyz: List[List[float]]) -> "np.ndarray":
#     import numpy as np
#     arr = np.array(acc_xyz, dtype="float32")
#
#     if THRESH_PAD > 0 and arr.shape[0] > THRESH_PAD:
#         arr = arr[THRESH_PAD:, :]
#
#
#     if arr.shape[0] >= INPUT_WINDOW:
#         arr = arr[-INPUT_WINDOW:, :]
#     else:
#         pad = np.zeros((INPUT_WINDOW - arr.shape[0], INPUT_DIM), dtype="float32")
#         arr = np.vstack([pad, arr])
#
#
#     arr = arr.reshape(1, INPUT_WINDOW, INPUT_DIM)
#     return arr
#
#
# def infer_label(arr: "np.ndarray") -> Tuple[str, float, int]:
#     import numpy as np
#     status, source = load_model_once()
#     if status != "ok":
#         return "ModelNotReady", 0.0, -1
#     probs = _model.predict(arr, verbose=0)[0]
#     idx = int(np.argmax(probs))
#     conf = float(probs[idx])
#     label = LABELS[idx] if 0 <= idx < len(LABELS) else f"class_{idx}"
#     return label, conf, idx
#
# @app.route("/", methods=["GET"])
# def home():
#     return "PDIoT Cloud API (model mode)", 200
#
# @app.route("/health", methods=["GET"])
# def health():
#     status, source = load_model_once()
#     return jsonify({
#         "ok": status == "ok",
#         "status": status,
#         "source": source,
#         "model_path": MODEL_PATH,
#         "model_url": MODEL_URL or None,
#         "input_window": INPUT_WINDOW,
#         "input_dim": INPUT_DIM,
#         "labels": LABELS,
#         "ts": int(time.time())
#     }), 200 if status == "ok" else 500
#
# @app.route("/schema", methods=["GET"])
# def schema():
#     return jsonify({
#         "accepted_json": {
#             "A_nested": {"acc": "[[x,y,z], ...]"},
#             "B_split_axes": {"x": "[...]", "y": "[...]", "z": "[...]"},
#             "C_timestamps": {"samples": "[{'time':t,'x':...,'y':...,'z':...}, ...]"}
#         },
#         "input_window": INPUT_WINDOW,
#         "input_dim": INPUT_DIM,
#         "labels": LABELS
#     }), 200
#
# @app.route("/classify", methods=["POST"])
# def classify():
#     t0 = time.time()
#     try:
#         payload = request.get_json(force=True, silent=False)
#         acc = parse_acc_data(payload)
#
#
#         if not acc or not isinstance(acc, list) or not isinstance(acc[0], list) or len(acc[0]) != INPUT_DIM:
#             app.logger.error(f"[ERROR] Bad JSON payload: {str(payload)[:200]} ...")
#             return jsonify({
#                 "ok": False,
#                 "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
#             }), 400
#
#         import numpy as np
#         arr = prepare_window(acc)
#         label, conf, idx = infer_label(arr)
#         latency_ms = int((time.time() - t0) * 1000)
#
#         # ---------- 调试日志：打印关键数据 ----------
#         acc_np = np.array(acc)
#         mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))
#         app.logger.info(
#             f"[INFO] Received window: {len(acc):3d} samples | "
#             f"mean magnitude: {mean_magnitude:.3f} | "
#             f"activity: {label:10s} | conf: {conf:.2f} | latency: {latency_ms} ms"
#         )
#
#
#         return jsonify({
#             "ok": True,
#             "activity": label,
#             "confidence": round(conf, 4),
#             "class_index": idx,
#             "window_used": min(len(acc), INPUT_WINDOW),
#             "latency_ms": latency_ms,
#             "labels": LABELS,
#             "ts": int(time.time())
#         }), 200
#
#     except Exception as e:
#         app.logger.exception(f"[EXCEPTION] classify() failed: {e}")
#         return jsonify({"ok": False, "error": str(e)}), 500
#
#
# if __name__ == "__main__":
#
#     app.run(host="0.0.0.0", port=5000, debug=True)
#
#
#

import os, time, json, hashlib, pathlib, tempfile
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# ========= 日志配置 =========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# ========= 模型配置 =========
MODEL_PATH = os.path.join("model", "task_1_activity_model_11_class.h5")
MODEL_URL    = ""
INPUT_WINDOW = int(os.getenv("INPUT_WINDOW", "25"))
INPUT_DIM    = int(os.getenv("INPUT_DIM", "3"))

LABELS = [
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

THRESH_PAD   = int(os.getenv("THRESH_PAD", "0"))
NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "false"

# ========= Flask 应用初始化 =========
app = Flask(__name__)
CORS(app)

# ========= 模型加载 =========
def _download_to(path: str, url: str) -> str:
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path

_model = None
_loaded_from = None

def load_model_once() -> Tuple[str, str]:
    global _model, _loaded_from
    if _model is not None:
        return "ok", _loaded_from

    model_path = MODEL_PATH
    if (not os.path.exists(model_path)) and MODEL_URL:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            _download_to(model_path, MODEL_URL)
            _loaded_from = "url"
        except Exception as e:
            app.logger.exception(f"Download model failed: {e}")
            return "error", "url"
    else:
        _loaded_from = "local" if os.path.exists(model_path) else "none"

    if not os.path.exists(model_path):
        return "missing", _loaded_from

    try:
        from tensorflow.keras.models import load_model
        _model = load_model(model_path, compile=False)
        return "ok", _loaded_from
    except Exception as e:
        app.logger.exception(f"Load model failed: {e}")
        return "error", _loaded_from

# ========= 数据解析与推理 =========
def parse_acc_data(payload: Dict[str, Any]) -> List[List[float]]:
    if isinstance(payload, dict) and "acc" in payload and isinstance(payload["acc"], list):
        return payload["acc"]
    if isinstance(payload, dict) and all(k in payload for k in ("x","y","z")):
        xs, ys, zs = payload["x"], payload["y"], payload["z"]
        if len(xs) == len(ys) == len(zs) and len(xs) > 0:
            return [[float(xs[i]), float(ys[i]), float(zs[i])] for i in range(len(xs))]
    if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
        out = []
        for s in payload["samples"]:
            if all(k in s for k in ("x","y","z")):
                out.append([float(s["x"]), float(s["y"]), float(s["z"])])
        return out
    return []

def prepare_window(acc_xyz: List[List[float]]) -> "np.ndarray":
    import numpy as np
    arr = np.array(acc_xyz, dtype="float32")
    if THRESH_PAD > 0 and arr.shape[0] > THRESH_PAD:
        arr = arr[THRESH_PAD:, :]
    if arr.shape[0] >= INPUT_WINDOW:
        arr = arr[-INPUT_WINDOW:, :]
    else:
        pad = np.zeros((INPUT_WINDOW - arr.shape[0], INPUT_DIM), dtype="float32")
        arr = np.vstack([pad, arr])
    arr = arr.reshape(1, INPUT_WINDOW, INPUT_DIM)
    return arr

def infer_label(arr: "np.ndarray") -> Tuple[str, float, int]:
    import numpy as np
    status, source = load_model_once()
    if status != "ok":
        return "ModelNotReady", 0.0, -1
    probs = _model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = LABELS[idx] if 0 <= idx < len(LABELS) else f"class_{idx}"
    return label, conf, idx

# ========= API 路由 =========
@app.route("/", methods=["GET"])
def home():
    return "PDIoT Cloud API (model mode)", 200

@app.route("/health", methods=["GET"])
def health():
    status, source = load_model_once()
    return jsonify({
        "ok": status == "ok",
        "status": status,
        "source": source,
        "model_path": MODEL_PATH,
        "model_url": MODEL_URL or None,
        "input_window": INPUT_WINDOW,
        "input_dim": INPUT_DIM,
        "labels": LABELS,
        "ts": int(time.time())
    }), 200 if status == "ok" else 500

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "accepted_json": {
            "A_nested": {"acc": "[[x,y,z], ...]"},
            "B_split_axes": {"x": "[...]", "y": "[...]", "z": "[...]"},
            "C_timestamps": {"samples": "[{'time':t,'x':...,'y':...,'z':...}, ...]"}
        },
        "input_window": INPUT_WINDOW,
        "input_dim": INPUT_DIM,
        "labels": LABELS
    }), 200

@app.route("/classify", methods=["POST"])
def classify():
    t0 = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
        acc = parse_acc_data(payload)

        if not acc or not isinstance(acc, list) or not isinstance(acc[0], list) or len(acc[0]) != INPUT_DIM:
            app.logger.error(f"[ERROR] Bad JSON payload: {str(payload)[:200]} ...")
            return jsonify({
                "ok": False,
                "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
            }), 400

        import numpy as np
        arr = prepare_window(acc)
        label, conf, idx = infer_label(arr)
        latency_ms = int((time.time() - t0) * 1000)

        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))
        app.logger.info(
            f"[INFO] Received window: {len(acc):3d} samples | "
            f"mean magnitude: {mean_magnitude:.3f} | "
            f"activity: {label:10s} | conf: {conf:.2f} | latency: {latency_ms} ms"
        )

        return jsonify({
            "ok": True,
            "activity": label,
            "confidence": round(conf, 4),
            "class_index": idx,
            "window_used": min(len(acc), INPUT_WINDOW),
            "latency_ms": latency_ms,
            "labels": LABELS,
            "ts": int(time.time())
        }), 200

    except Exception as e:
        app.logger.exception(f"[EXCEPTION] classify() failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ==========================
# 新增：活动日志模块
# ==========================
from collections import defaultdict

DATA_FILE = "activity_history.json"  # 日志存储路径

def load_history() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_history(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/log_activity", methods=["POST"])
def log_activity():
    """
    前端每次分类后调用此接口保存活动记录。
    当天累积；若进入新日期，则自动重置。
    """
    try:
        payload = request.get_json(force=True)
        user_id = payload.get("user_id", "anonymous")
        activity = payload.get("activity", "unknown")
        confidence = float(payload.get("confidence", 0))
        ts = int(payload.get("timestamp", time.time()))
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))

        # ========== 加载历史 ==========
        history = load_history()
        user_hist = history.setdefault(user_id, {})

        # ✅ 如果已有记录，但不是今天的，自动清零
        if user_hist and date_str not in user_hist:
            app.logger.info(f"[RESET] New day detected for {user_id}. Clearing old history.")
            user_hist.clear()

        # ========== 累积逻辑 ==========
        day_hist = user_hist.setdefault(date_str, defaultdict(float))

        # 每窗口 2 秒（根据采样率12.5Hz、窗口25）
        WINDOW_DURATION = INPUT_WINDOW / 12.5
        if "cough" in activity.lower():
            day_hist[activity] = round(day_hist.get(activity, 0.0) + 1, 2)
        else:
            day_hist[activity] = round(day_hist.get(activity, 0.0) + WINDOW_DURATION, 2)

        # 保存结果
        history[user_id][date_str] = dict(day_hist)
        save_history(history)

        app.logger.info(f"[LOG] {user_id} -> {activity} (+{WINDOW_DURATION}s) on {date_str}")
        return jsonify({"ok": True, "msg": f"Logged {activity} for {user_id}"}), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify({"ok": False, "error": str(e)}), 500




@app.route("/history/<user>/<date>", methods=["GET"])
def get_history(user, date):
    """
    返回某用户在某日的活动统计。
    例如 GET /history/user1/2025-11-04
    """
    history = load_history()
    user_hist = history.get(user, {})
    result = user_hist.get(date, {})
    return jsonify({"user": user, "date": date, "summary": result}), 200


# ========= 程序入口 =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

