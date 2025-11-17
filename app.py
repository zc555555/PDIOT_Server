import os, time, json, hashlib, pathlib, tempfile
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import tensorflow as tf

# ========= 日志配置 =========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# ========= 模型1配置：daily activity =========
MODEL_PATH = os.path.join("model", "human_activity.h5")   # ⭐ 新的活动模型
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

# static 状态，只在这些状态下跑 social signal 模型
STATIC_ACTIVITIES = {
    "sittingStanding",
    "lyingLeft",
    "lyingRight",
    "lyingBack",
    "lyingStomach"
}

THRESH_PAD   = int(os.getenv("THRESH_PAD", "0"))
NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "false"  # 原来的逻辑，先留着

# ========= 模型2配置：social signal =========
SOCIAL_MODEL_PATH = os.path.join("model", "social_signal_cnn_lstm.h5")
SOCIAL_MODEL_URL = ""
SOCIAL_INPUT_WINDOW = int(os.getenv("SOCIAL_INPUT_WINDOW", "50"))
SOCIAL_INPUT_DIM = INPUT_DIM  # 3 维加速度

# social signal 四分类标签（来自你的训练脚本）
SOCIAL_CLASS_NAMES = [
    "breathingNormally",
    "coughing",
    "hyperventilation",
    "other"
]

# 训练时使用的 mean / std（直接硬编码到服务端，保证和训练一致）
SOCIAL_NORM_MEAN = [-0.07304631, -0.3672524, 0.06649867]
SOCIAL_NORM_STD  = [0.5281695, 0.44663346, 0.6301666]

# ========= Flask 应用初始化 =========
app = Flask(__name__)
CORS(app)

# ========= 模型1加载（daily activity） =========
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
    """
    加载 daily activity 模型（human_activity.h5）
    """
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
        # from tensorflow.keras.models import load_model
        # _model = load_model(model_path, compile=False)
        _model = tf.keras.models.load_model(model_path, compile=False)
        return "ok", _loaded_from
    except Exception as e:
        app.logger.exception(f"Load model failed: {e}")
        return "error", _loaded_from

# ========= 模型2加载（social signal） =========
_social_model = None
_social_loaded_from = None
_social_labels = None

def load_social_model_once() -> Tuple[str, str]:
    """
    加载 social signal 模型（social_signal_cnn_lstm.h5）
    """
    global _social_model, _social_loaded_from, _social_labels
    if _social_model is not None:
        return "ok", _social_loaded_from

    model_path = SOCIAL_MODEL_PATH
    if (not os.path.exists(model_path)) and SOCIAL_MODEL_URL:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            _download_to(model_path, SOCIAL_MODEL_URL)
            _social_loaded_from = "url"
        except Exception as e:
            app.logger.exception(f"Download social model failed: {e}")
            return "error", "url"
    else:
        _social_loaded_from = "local" if os.path.exists(model_path) else "none"

    if not os.path.exists(model_path):
        return "missing", _social_loaded_from

    try:
        # from tensorflow.keras.models import load_model
        # _social_model = load_model(model_path, compile=False)

        _social_model = tf.keras.models.load_model(model_path, compile=False)

        # 检查输出维度是否和 4 类一致
        try:
            n_classes = _social_model.output_shape[-1]
            if n_classes != len(SOCIAL_CLASS_NAMES):
                app.logger.warning(
                    f"[WARN] social model output classes = {n_classes}, "
                    f"but SOCIAL_CLASS_NAMES has {len(SOCIAL_CLASS_NAMES)} labels."
                )
        except Exception:
            pass

        _social_labels = SOCIAL_CLASS_NAMES
        return "ok", _social_loaded_from
    except Exception as e:
        app.logger.exception(f"Load social model failed: {e}")
        return "error", _social_loaded_from

# ========= 数据解析 =========
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

# ========= 窗口准备 =========
def prepare_window_activity(acc_xyz: List[List[float]]) -> "np.ndarray":
    """
    给 daily activity 模型用的窗口（25x3）
    """
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

def prepare_window_social(acc_xyz: List[List[float]]) -> "np.ndarray":
    """
    给 social signal 模型用的窗口（50x3）+ 标准化（用训练时的 mean/std）
    """
    import numpy as np
    arr = np.array(acc_xyz, dtype="float32")

    # 截/补成 50x3
    if arr.shape[0] >= SOCIAL_INPUT_WINDOW:
        arr = arr[-SOCIAL_INPUT_WINDOW:, :]
    else:
        pad = np.zeros((SOCIAL_INPUT_WINDOW - arr.shape[0], SOCIAL_INPUT_DIM), dtype="float32")
        arr = np.vstack([pad, arr])

    # 用训练的 mean / std 做 z-score 标准化
    mean = np.array(SOCIAL_NORM_MEAN, dtype="float32")
    std = np.array(SOCIAL_NORM_STD, dtype="float32")
    arr = (arr - mean) / std

    arr = arr.reshape(1, SOCIAL_INPUT_WINDOW, SOCIAL_INPUT_DIM)
    return arr

# ========= 推理 =========
def infer_label(arr: "np.ndarray") -> Tuple[str, float, int]:
    """
    daily activity 推理
    """
    import numpy as np
    status, source = load_model_once()
    if status != "ok":
        return "ModelNotReady", 0.0, -1
    probs = _model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = LABELS[idx] if 0 <= idx < len(LABELS) else f"class_{idx}"
    return label, conf, idx

def infer_social_label(arr: "np.ndarray") -> Tuple[str, float, int]:
    """
    social signal 推理
    """
    import numpy as np
    status, source = load_social_model_once()
    if status != "ok":
        return "SocialModelNotReady", 0.0, -1
    probs = _social_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    global _social_labels
    if not _social_labels or idx >= len(_social_labels):
        label = f"class_{idx}"
    else:
        label = _social_labels[idx]
    return label, conf, idx

# ========= API 路由 =========
@app.route("/", methods=["GET"])
def home():
    return "PDIoT Cloud API (model mode)", 200

@app.route("/health", methods=["GET"])
def health():
    status1, source1 = load_model_once()
    status2, source2 = load_social_model_once()
    return jsonify({
        "ok": (status1 == "ok"),
        "activity_model": {
            "ok": status1 == "ok",
            "status": status1,
            "source": source1,
            "model_path": MODEL_PATH,
            "input_window": INPUT_WINDOW,
            "input_dim": INPUT_DIM,
            "labels": LABELS,
        },
        "social_model": {
            "ok": status2 == "ok",
            "status": status2,
            "source": source2,
            "model_path": SOCIAL_MODEL_PATH,
            "input_window": SOCIAL_INPUT_WINDOW,
            "input_dim": SOCIAL_INPUT_DIM,
            "labels": _social_labels,
            "norm_mean": SOCIAL_NORM_MEAN,
            "norm_std": SOCIAL_NORM_STD,
        },
        "ts": int(time.time())
    }), 200 if status1 == "ok" else 500

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "accepted_json": {
            "A_nested": {"acc": "[[x,y,z], ...]"},
            "B_split_axes": {"x": "[...]", "y": "[...]", "z": "[...]"},
            "C_timestamps": {"samples": "[{'time':t,'x':...,'y':...,'z':...}, ...]"}
        },
        "activity_model": {
            "input_window": INPUT_WINDOW,
            "input_dim": INPUT_DIM,
            "labels": LABELS
        },
        "social_model": {
            "input_window": SOCIAL_INPUT_WINDOW,
            "input_dim": SOCIAL_INPUT_DIM,
            "labels": _social_labels or SOCIAL_CLASS_NAMES
        },
        "static_activities_for_social": list(STATIC_ACTIVITIES)
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
        # ===== 1. daily activity 预测 =====
        arr_activity = prepare_window_activity(acc)
        activity_label, activity_conf, activity_idx = infer_label(arr_activity)
        latency_ms = int((time.time() - t0) * 1000)

        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))

        # ===== 2. 若是 static 状态，再跑 social signal =====
        social_result = None
        if activity_label in STATIC_ACTIVITIES:
            arr_social = prepare_window_social(acc)
            social_label, social_conf, social_idx = infer_social_label(arr_social)
            if social_idx >= 0:
                social_result = {
                    "label": social_label,
                    "confidence": round(social_conf, 4),
                    "class_index": social_idx
                }
            else:
                social_result = {
                    "error": "Social model not ready"
                }

            app.logger.info(
                f"[INFO] Social signal | activity: {activity_label} | "
                f"social: {social_label} | conf: {social_conf:.2f}"
            )

        app.logger.info(
            f"[INFO] Received window: {len(acc):3d} samples | "
            f"mean magnitude: {mean_magnitude:.3f} | "
            f"activity: {activity_label:10s} | conf: {activity_conf:.2f} | "
            f"latency: {latency_ms} ms"
        )

        return jsonify({
            "ok": True,
            "activity": activity_label,
            "confidence": round(activity_conf, 4),
            "class_index": activity_idx,
            "window_used": min(len(acc), INPUT_WINDOW),
            "latency_ms": latency_ms,
            "labels": LABELS,
            "social_signal": social_result,   # ⭐ 静态才有结果，否则为 null
            "ts": int(time.time())
        }), 200

    except Exception as e:
        app.logger.exception(f"[EXCEPTION] classify() failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ==========================
# 活动日志模块（原样保留）
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


@app.route("/classify_activity", methods=["POST"])
def classify_activity():
    """
    只做 daily activity 分类，输入窗口为 25x3
    前端可以直接传 25x3，也可以更长，后端会取最后 25 个点，不够则补零
    """
    t0 = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
        acc = parse_acc_data(payload)

        if not acc or not isinstance(acc, list) or not isinstance(acc[0], list) or len(acc[0]) != INPUT_DIM:
            app.logger.error(f"[ERROR] Bad JSON payload (activity): {str(payload)[:200]} ...")
            return jsonify({
                "ok": False,
                "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
            }), 400

        import numpy as np
        arr = prepare_window_activity(acc)
        label, conf, idx = infer_label(arr)
        latency_ms = int((time.time() - t0) * 1000)

        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))

        app.logger.info(
            f"[ACTIVITY] samples={len(acc):3d} | mean_mag={mean_magnitude:.3f} | "
            f"activity={label} | conf={conf:.2f} | latency={latency_ms} ms"
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
        app.logger.exception(f"[EXCEPTION] classify_activity() failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/classify_social", methods=["POST"])
def classify_social():
    """
    只做 social signal 分类，输入窗口为 50x3
    默认前端只在 static activity (sitting / lying*) 时调用。
    可选参数: activity，用来做一个简单校验。
    """
    t0 = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
        acc = parse_acc_data(payload)
        activity_from_frontend = payload.get("activity")  # 可选

        if not acc or not isinstance(acc, list) or not isinstance(acc[0], list) or len(acc[0]) != SOCIAL_INPUT_DIM:
            app.logger.error(f"[ERROR] Bad JSON payload (social): {str(payload)[:200]} ...")
            return jsonify({
                "ok": False,
                "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
            }), 400

        # 如果前端传了 activity，而且不是 static，可以直接拒绝（可选逻辑）
        if activity_from_frontend is not None and activity_from_frontend not in STATIC_ACTIVITIES:
            return jsonify({
                "ok": False,
                "error": f"Activity '{activity_from_frontend}' is not static, social signal not evaluated.",
                "allowed_static_activities": list(STATIC_ACTIVITIES)
            }), 400

        arr_social = prepare_window_social(acc)
        label, conf, idx = infer_social_label(arr_social)
        latency_ms = int((time.time() - t0) * 1000)

        app.logger.info(
            f"[SOCIAL] samples={len(acc):3d} | social={label} | conf={conf:.2f} | "
            f"latency={latency_ms} ms | activity_hint={activity_from_frontend}"
        )

        return jsonify({
            "ok": True,
            "social_label": label,
            "social_confidence": round(conf, 4),
            "social_class_index": idx,
            "window_used": min(len(acc), SOCIAL_INPUT_WINDOW),
            "latency_ms": latency_ms,
            "labels": _social_labels or SOCIAL_CLASS_NAMES,
            "ts": int(time.time())
        }), 200

    except Exception as e:
        app.logger.exception(f"[EXCEPTION] classify_social() failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500



# ========= 程序入口 =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
