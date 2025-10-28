import os, time, json, hashlib, pathlib, tempfile
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS

# ========= 配置（用环境变量覆盖） =========
MODEL_PATH   = r"C:\Users\13785\OneDrive\Desktop\pdiot_cw3\model\task_1_activity_model_11_class.h5"
# MODEL_PATH   = r"C:\Users\13785\OneDrive\Desktop\pdiot_cw3\model\har_model.h5"
MODEL_URL    = ""   # 不再从网络下载
INPUT_WINDOW = int(os.getenv("INPUT_WINDOW", "25"))           # 模型输入时间步
INPUT_DIM    = int(os.getenv("INPUT_DIM", "3"))                # 通道数（x,y,z）
# LABELS = os.getenv(
#     "LABELS",
#     "ascending,descending,lyingBack,lyingLeft,lyingRight,lyingStomach,miscMovement,normalWalking,running,shuffleWalking,sittingStanding"
# ).split(",")

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


THRESH_PAD   = int(os.getenv("THRESH_PAD", "0"))               # 可选前置丢帧
# NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "true"
NORMALIZE = os.getenv("NORMALIZE", "true").lower() == "false"


app = Flask(__name__)
CORS(app)  # 允许跨域（防止前端请求被拦）

# ========= 工具：下载模型（若提供URL）=========
def _download_to(path: str, url: str) -> str:
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path

# ========= 模型加载 =========
_model = None
_loaded_from = None
def load_model_once() -> Tuple[str, str]:
    """
    返回 (status, source)
    status: 'ok'|'missing'|'error'
    source: 'local'|'url'|'none'
    """
    global _model, _loaded_from
    if _model is not None:
        return "ok", _loaded_from

    model_path = MODEL_PATH
    # 若本地不存在且提供了 URL，则下载
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
        # 延迟导入，避免部署前安装冲突
        from tensorflow.keras.models import load_model
        # _model = load_model(model_path)
        _model = load_model(model_path, compile=False)

        return "ok", _loaded_from
    except Exception as e:
        app.logger.exception(f"Load model failed: {e}")
        return "error", _loaded_from

# def load_model_once() -> Tuple[str, str]:
#     global _model, _loaded_from
#     if _model is not None:
#         return "ok", _loaded_from
#
#     model_path = MODEL_PATH
#     if not os.path.exists(model_path):
#         return "missing", "none"
#
#     from tensorflow.keras.models import load_model
#     try:
#         # ✅ 优先尝试新版Keras加载
#         _model = load_model(model_path, compile=False)
#         _loaded_from = "local"
#         return "ok", _loaded_from
#     except Exception as e1:
#         app.logger.warning(f"Primary load failed: {e1}, retrying with safe_mode=False ...")
#         try:
#             # ✅ 尝试兼容模式加载（旧版模型）
#             _model = load_model(model_path, compile=False, safe_mode=False)
#             _loaded_from = "compat"
#             return "ok", _loaded_from
#         except Exception as e2:
#             app.logger.exception(f"Both load attempts failed: {e2}")
#             return "error", "local"



# ========= JSON 解析（兼容三类格式）=========
def parse_acc_data(payload: Dict[str, Any]) -> List[List[float]]:
    # A) {"acc": [[x,y,z], ...]}
    if isinstance(payload, dict) and "acc" in payload and isinstance(payload["acc"], list):
        return payload["acc"]
    # B) {"x":[...],"y":[...],"z":[...]}
    if isinstance(payload, dict) and all(k in payload for k in ("x","y","z")):
        xs, ys, zs = payload["x"], payload["y"], payload["z"]
        if len(xs) == len(ys) == len(zs) and len(xs) > 0:
            return [[float(xs[i]), float(ys[i]), float(zs[i])] for i in range(len(xs))]
    # C) {"samples":[{"time":t,"x":...,"y":...,"z":...}, ...]}
    if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
        out = []
        for s in payload["samples"]:
            if all(k in s for k in ("x","y","z")):
                out.append([float(s["x"]), float(s["y"]), float(s["z"])])
        return out
    return []

# ========= 预处理（截断/补零/标准化）=========
def prepare_window(acc_xyz: List[List[float]]) -> "np.ndarray":
    import numpy as np
    arr = np.array(acc_xyz, dtype="float32")
    # 可选：丢弃前面若干不稳定帧（蓝牙拼包/冷启动）
    if THRESH_PAD > 0 and arr.shape[0] > THRESH_PAD:
        arr = arr[THRESH_PAD:, :]

    # 对齐长度到 INPUT_WINDOW
    if arr.shape[0] >= INPUT_WINDOW:
        arr = arr[-INPUT_WINDOW:, :]
    else:
        pad = np.zeros((INPUT_WINDOW - arr.shape[0], INPUT_DIM), dtype="float32")
        arr = np.vstack([pad, arr])

    # if NORMALIZE:
    #     # 简单标准化：逐通道零均值单位方差（避免除零）
    #     mean = arr.mean(axis=0, keepdims=True)
    #     std  = arr.std(axis=0, keepdims=True) + 1e-6
    #     arr  = (arr - mean) / std

    arr = arr.reshape(1, INPUT_WINDOW, INPUT_DIM)
    return arr

# ========= 推理 =========
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

# ========= 路由 =========
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

        # ---------- 调试日志：检查输入 ----------
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

        # ---------- 调试日志：打印关键数据 ----------
        acc_np = np.array(acc)
        mean_magnitude = float(np.mean(np.sqrt(np.sum(acc_np**2, axis=1))))
        app.logger.info(
            f"[INFO] Received window: {len(acc):3d} samples | "
            f"mean magnitude: {mean_magnitude:.3f} | "
            f"activity: {label:10s} | conf: {conf:.2f} | latency: {latency_ms} ms"
        )

        # ---------- 返回结果 ----------
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


if __name__ == "__main__":
    # 本地开发：设置 MODEL_PATH 或 MODEL_URL 后运行
    app.run(host="0.0.0.0", port=5000, debug=True)



