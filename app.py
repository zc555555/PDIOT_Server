import os, time, json, hashlib, pathlib, tempfile
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import tensorflow as tf
from collections import defaultdict, deque  # ⭐ 新增 deque，用于滚动窗口缓冲

# ========= 日志配置 =========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# ========= 模型1配置：daily activity =========
# 使用你最新训练的 11-class 模型 (TIMESTEPS = 50)
MODEL_PATH = os.path.join("model", "task_1_activity_model_11_class.h5")
MODEL_URL    = ""

# ⭐ 窗口长度 50，和训练脚本 TIMESTEPS = 50 一致
INPUT_WINDOW = int(os.getenv("INPUT_WINDOW", "50"))
INPUT_DIM    = int(os.getenv("INPUT_DIM", "3"))

# ⭐ 标签顺序必须和训练脚本 CLASSES_MAP 完全一致
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

# 方便做后处理：名字 -> index
ACTIVITY_MAP = {name: idx for idx, name in enumerate(LABELS)}

# static 状态，只在这些状态下跑 social signal 模型
STATIC_ACTIVITIES = {
    "sittingStanding",
    "lyingLeft",
    "lyingRight",
    "lyingBack",
    "lyingStomach"
}

THRESH_PAD   = int(os.getenv("THRESH_PAD", "0"))
NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "false"  # 保留原逻辑

# ========= 模型2配置：social signal =========
SOCIAL_MODEL_PATH = os.path.join("model", "social_signal_cnn_lstm.h5")
SOCIAL_MODEL_URL = ""
SOCIAL_INPUT_WINDOW = int(os.getenv("SOCIAL_INPUT_WINDOW", "50"))
SOCIAL_INPUT_DIM = INPUT_DIM  # 3 维加速度

SOCIAL_CLASS_NAMES = [
    "breathingNormally",
    "coughing",
    "hyperventilation",
    "other"
]

# 训练时使用的 mean / std（直接硬编码到服务端，保证和训练一致）
SOCIAL_NORM_MEAN = [-0.07304631, -0.3672524, 0.06649867]
SOCIAL_NORM_STD  = [0.5281695, 0.44663346, 0.6301666]

# ========= 全局缓冲：用于在线滚动窗口 =========
# ⭐ 关键：服务器端维护最近 50 个样本，保证送入模型的是“连续 50 点”
ACC_BUFFER = deque(maxlen=INPUT_WINDOW)

# ========= Flask 应用初始化 =========
app = Flask(__name__)
CORS(app)

# ========= 下载工具 =========
def _download_to(path: str, url: str) -> str:
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path

# ========= 模型1加载（daily activity） =========
_model = None
_loaded_from = None

def load_model_once() -> Tuple[str, str]:
    """
    加载 daily activity 模型（task_1_activity_model_11_class.h5）
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
    给 daily activity 模型用的窗口（50x3）
    这里假定 acc_xyz 已经是“尽可能长的连续序列”，会取最后 50 个点，不够前面补 0。
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
    arr = (arr - mean) / (std + 1e-8)

    arr = arr.reshape(1, SOCIAL_INPUT_WINDOW, SOCIAL_INPUT_DIM)
    return arr

# ========= daily activity 概率后处理（与你的 test 脚本一致） =========
def _postprocess_activity_probs(probs_raw: "np.ndarray") -> Tuple[int, float]:
    """
    对单个窗口的 11 维概率做：
    1) alpha 重权
    2) 规则修正 (ascending vs normal, sitting/descending vs shuffle, shuffle vs misc)
    返回: (final_idx, final_conf)
    """
    import numpy as np

    num_classes = len(LABELS)
    probs = probs_raw.astype("float32")

    alpha = np.ones(num_classes, dtype=np.float32)

    sitting_idx    = ACTIVITY_MAP["sittingStanding"]
    normal_idx     = ACTIVITY_MAP["normalWalking"]
    shuffle_idx    = ACTIVITY_MAP["shuffleWalking"]
    misc_idx       = ACTIVITY_MAP["miscMovement"]
    ascending_idx  = ACTIVITY_MAP["ascending"]
    descending_idx = ACTIVITY_MAP["descending"]

    # 压一些 dominating 类
    alpha[sitting_idx]    = 0.80
    alpha[ascending_idx]  = 0.90
    alpha[descending_idx] = 0.90

    # 抬高 hard 类
    alpha[normal_idx]  = 1.10
    alpha[shuffle_idx] = 1.30
    alpha[misc_idx]    = 1.10

    probs_scaled = probs * alpha
    probs_scaled = probs_scaled / (probs_scaled.sum() + 1e-8)

    pred = int(np.argmax(probs_scaled))

    margin_asc_norm    = 0.10
    margin_shuffle_cut = 0.10

    p_norm    = float(probs_scaled[normal_idx])
    p_asc     = float(probs_scaled[ascending_idx])
    p_shuffle = float(probs_scaled[shuffle_idx])
    p_sit     = float(probs_scaled[sitting_idx])
    p_misc    = float(probs_scaled[misc_idx])
    p_desc    = float(probs_scaled[descending_idx])

    # 1) ascending vs normalWalking 修正
    if pred == ascending_idx and (p_asc - p_norm) < margin_asc_norm:
        pred = normal_idx

    # 2) sitting → shuffle
    if pred == sitting_idx and (p_shuffle - p_sit) > -margin_shuffle_cut:
        pred = shuffle_idx

    # 3) descending → shuffle
    if pred == descending_idx and (p_shuffle - p_desc) > -margin_shuffle_cut:
        pred = shuffle_idx

    # 4) shuffle → misc
    if pred == shuffle_idx and (p_misc - p_shuffle) > margin_shuffle_cut:
        pred = misc_idx

    final_conf = float(probs_scaled[pred])
    return pred, final_conf

# ========= 推理 =========
def infer_label(arr: "np.ndarray") -> Tuple[str, float, int]:
    """
    daily activity 推理（用 task_1_activity_model_11_class.h5 + alpha/规则后处理）
    """
    import numpy as np
    status, source = load_model_once()
    if status != "ok":
        return "ModelNotReady", 0.0, -1

    probs = _model.predict(arr, verbose=0)[0]  # shape (11,)
    idx, conf = _postprocess_activity_probs(probs)

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
    """
    综合接口：先做 daily activity，如果是静态类，再做 social signal。
    期望 acc 形状为 [N, 3]，前端可以每次发一小段，后端会累积到 50 个样本。
    """
    t0 = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
        acc = parse_acc_data(payload)

        if (not acc or not isinstance(acc, list) or
                not isinstance(acc[0], list) or len(acc[0]) != INPUT_DIM):
            app.logger.error(f"[ERROR] Bad JSON payload: {str(payload)[:200]} ...")
            return jsonify({
                "ok": False,
                "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
            }), 400

        import numpy as np

        # ⭐ 关键：累积到全局 ACC_BUFFER，保证用连续 50 点预测
        for s in acc:
            ACC_BUFFER.append(s)

        buffer_len = len(ACC_BUFFER)
        arr_activity = prepare_window_activity(list(ACC_BUFFER))
        activity_label, activity_conf, activity_idx = infer_label(arr_activity)
        latency_ms = int((time.time() - t0) * 1000)

        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))

        # ===== 2. 若是 static 状态，再跑 social signal（仍使用本次 acc） =====
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
            f"buffer_len: {buffer_len:3d} | "
            f"mean magnitude: {mean_magnitude:.3f} | "
            f"activity: {activity_label:15s} | conf: {activity_conf:.2f} | "
            f"latency: {latency_ms} ms"
        )

        return jsonify({
            "ok": True,
            "activity": activity_label,
            "confidence": round(activity_conf, 4),
            "class_index": activity_idx,
            "window_used": min(buffer_len, INPUT_WINDOW),
            "latency_ms": latency_ms,
            "labels": LABELS,
            "social_signal": social_result,
            "ts": int(time.time())
        }), 200

    except Exception as e:
        app.logger.exception(f"[EXCEPTION] classify() failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# ==========================
# 活动日志模块（原样保留）
# ==========================
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

        # 每窗口的时间长度 = INPUT_WINDOW / 12.5
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
    只做 daily activity 分类。
    前端可以每次发 N 个样本（例如 25），后端会累积到 50 个，送入模型。
    """
    t0 = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
        acc = parse_acc_data(payload)

        if (not acc or not isinstance(acc, list) or
                not isinstance(acc[0], list) or len(acc[0]) != INPUT_DIM):
            app.logger.error(f"[ERROR] Bad JSON payload (activity): {str(payload)[:200]} ...")
            return jsonify({
                "ok": False,
                "error": "Bad JSON: expected accelerometer array shape [N, 3]. See /schema"
            }), 400

        import numpy as np

        # ⭐ 同样累积到 ACC_BUFFER
        for s in acc:
            ACC_BUFFER.append(s)

        buffer_len = len(ACC_BUFFER)
        arr = prepare_window_activity(list(ACC_BUFFER))
        label, conf, idx = infer_label(arr)
        latency_ms = int((time.time() - t0) * 1000)

        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))

        app.logger.info(
            f"[ACTIVITY] samples={len(acc):3d} | buffer_len={buffer_len:3d} | "
            f"mean_mag={mean_magnitude:.3f} | activity={label} | "
            f"conf={conf:.2f} | latency={latency_ms} ms"
        )

        return jsonify({
            "ok": True,
            "activity": label,
            "confidence": round(conf, 4),
            "class_index": idx,
            "window_used": min(buffer_len, INPUT_WINDOW),
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

        if (not acc or not isinstance(acc, list) or
                not isinstance(acc[0], list) or len(acc[0]) != SOCIAL_INPUT_DIM):
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
