import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymysql
import torch
import torch.nn as nn
from flask import Flask, jsonify, request

from config import APP_CONFIG, DB_CONFIG, MODEL_CONFIG, THRESHOLD_DEFAULTS

app = Flask(__name__)

# =========================
# 全局缓存：CNN 模型
# =========================
_CNN_MODEL = None
_CNN_MODEL_META = {
    "loaded": False,
    "checkpoint_path": MODEL_CONFIG["cnn_checkpoint_path"],
    "model_name": MODEL_CONFIG["cnn_model_name"],
    "input_shape": [2, MODEL_CONFIG["cnn_input_length"]],
    "label_map": MODEL_CONFIG["label_map"],
    "idx_to_name": {v: k for k, v in MODEL_CONFIG["label_map"].items()},
    "error": None,
}


# =========================
# 通用返回
# =========================
def success(data: Any = None, msg: str = "操作成功"):
    return jsonify({
        "code": 200,
        "msg": msg,
        "data": data
    })


def fail(code: int, msg: str):
    return jsonify({
        "code": code,
        "msg": msg,
        "data": None
    }), code


# =========================
# 数据库与阈值
# =========================
def get_db_connection():
    return pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def load_thresholds() -> Tuple[Dict[str, float], str]:
    threshold_map = {
        "alarm.power.threshold.dbm": THRESHOLD_DEFAULTS["alarm.power.threshold.dbm"],
        "alarm.snr.threshold.db": THRESHOLD_DEFAULTS["alarm.snr.threshold.db"]
    }

    source = "default"

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT config_key, config_value
            FROM sys_config
            WHERE config_key IN ('alarm.power.threshold.dbm', 'alarm.snr.threshold.db')
            """
            cursor.execute(sql)
            rows = cursor.fetchall()

            for row in rows:
                key = row.get("config_key")
                value = row.get("config_value")
                if key in threshold_map:
                    try:
                        threshold_map[key] = float(value)
                    except Exception:
                        pass

            source = "sys_config"
    except Exception as e:
        print(f"[WARN] 读取 sys_config 阈值失败，改用默认值。error={e}")
    finally:
        if conn:
            conn.close()

    return threshold_map, source


# =========================
# 规则模型相关
# =========================
def normalize_points(points: Any) -> List[float]:
    if not isinstance(points, list):
        return []

    result = []
    for item in points:
        try:
            result.append(float(item))
        except Exception:
            continue
    return result


def estimate_noise_floor(points: List[float]) -> float:
    if not points:
        return -90.0

    sorted_points = sorted(points)
    take_n = max(1, int(len(points) * 0.2))
    return round(sum(sorted_points[:take_n]) / take_n, 2)


def estimate_active_width(points: List[float], threshold: float) -> int:
    return sum(1 for x in points if x >= threshold)


def predict_rule(data: Dict[str, Any], threshold_map: Dict[str, float], threshold_source: str) -> Dict[str, Any]:
    center_freq = float(data.get("center_freq_mhz", 0))
    bandwidth_khz = float(data.get("bandwidth_khz", 0))
    peak_power_dbm = float(data.get("peak_power_dbm", -90))
    snr_db = float(data.get("snr_db", 0))
    occupied_bw = float(data.get("occupied_bandwidth_khz", 0))
    channel_model = str(data.get("channel_model", "UNKNOWN"))
    power_points = normalize_points(data.get("power_points", []))

    power_threshold = float(threshold_map["alarm.power.threshold.dbm"])
    snr_threshold = float(threshold_map["alarm.snr.threshold.db"])

    if not power_points:
        return {
            "predicted_label": "UNKNOWN",
            "confidence": 0.50,
            "risk_level": "MEDIUM",
            "should_alarm": False,
            "reason": "未提供有效频谱采样点，无法做稳定识别。",
            "model_name": MODEL_CONFIG["rule_model_name"],
            "inference_mode": "rule",
            "actual_mode": "rule",
            "fallback_used": False,
            "fallback_reason": "",
            "thresholds": {
                "power_alarm_threshold_dbm": power_threshold,
                "snr_alarm_threshold_db": snr_threshold,
                "source": threshold_source
            }
        }

    noise_floor = estimate_noise_floor(power_points)
    active_threshold = noise_floor + 3.0
    active_bins = estimate_active_width(power_points, active_threshold)

    if occupied_bw <= 0 and bandwidth_khz > 0 and len(power_points) > 0:
        occupied_bw = round(active_bins * (bandwidth_khz / len(power_points)), 2)

    if center_freq < 150:
        if occupied_bw >= 170:
            predicted_label = "FM"
            reason = "处于低频模拟频段，且占用带宽较宽，更接近 FM 特征。"
            confidence = 0.86
        else:
            predicted_label = "AM"
            reason = "处于低频模拟频段，且占用带宽较窄，更接近 AM 特征。"
            confidence = 0.82
    else:
        if occupied_bw < 180:
            predicted_label = "BPSK"
            reason = "处于数字调制频段，占用带宽较窄，更接近 BPSK。"
            confidence = 0.79
        elif occupied_bw < 230:
            predicted_label = "QPSK"
            reason = "处于数字调制频段，占用带宽中等，更接近 QPSK。"
            confidence = 0.83
        else:
            predicted_label = "16QAM"
            reason = "处于数字调制频段，占用带宽较宽，更接近 16QAM。"
            confidence = 0.85

    if channel_model in ("Rayleigh", "CarrierOffset", "SampleRateError"):
        confidence = max(0.60, round(confidence - 0.06, 2))
    else:
        confidence = round(confidence, 2)

    should_alarm = (peak_power_dbm >= power_threshold) or (snr_db <= snr_threshold)

    if peak_power_dbm >= power_threshold + 5 or snr_db <= snr_threshold - 3:
        risk_level = "HIGH"
    elif should_alarm:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    extra_reason = (
        f" 峰值功率={peak_power_dbm}dBm，SNR={snr_db}dB，"
        f"估计底噪={noise_floor}dBm，活跃频点={active_bins}。"
    )

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "risk_level": risk_level,
        "should_alarm": should_alarm,
        "reason": reason + extra_reason,
        "model_name": MODEL_CONFIG["rule_model_name"],
        "inference_mode": "rule",
        "actual_mode": "rule",
        "fallback_used": False,
        "fallback_reason": "",
        "thresholds": {
            "power_alarm_threshold_dbm": power_threshold,
            "snr_alarm_threshold_db": snr_threshold,
            "source": threshold_source
        }
    }


# =========================
# 1D-CNN 模型定义
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class OneDCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(2, 32, kernel_size=7, pool=True),
            ConvBlock(32, 64, kernel_size=5, pool=True),
            ConvBlock(64, 128, kernel_size=3, pool=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# CNN 模型加载与预处理
# =========================
def get_compute_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_iq_per_channel(iq: np.ndarray) -> np.ndarray:
    iq = iq.astype(np.float32).copy()
    for c in range(iq.shape[0]):
        mean = iq[c].mean()
        std = iq[c].std() + 1e-6
        iq[c] = (iq[c] - mean) / std
    return iq


def resize_iq_to_target_length(iq: np.ndarray, target_len: int) -> np.ndarray:
    """
    将 [2, L] 插值到 [2, target_len]
    """
    if iq.shape[1] == target_len:
        return iq.astype(np.float32)

    old_idx = np.linspace(0, 1, iq.shape[1])
    new_idx = np.linspace(0, 1, target_len)

    out = np.zeros((2, target_len), dtype=np.float32)
    out[0] = np.interp(new_idx, old_idx, iq[0]).astype(np.float32)
    out[1] = np.interp(new_idx, old_idx, iq[1]).astype(np.float32)
    return out



def parse_numeric_sequence(value: Any, field_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    把输入安全转换成一维 float32 数组。
    允许：
    1. 原生 list / tuple
    2. numpy.ndarray
    3. JSON 字符串形式的数组，例如 "[0.1, 0.2]"
    不允许：
    1. 单个标量
    2. 空字符串
    3. 维度为 0 的 numpy 标量数组
    """
    if value is None:
        return None, f"字段 {field_name} 不能为空"

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None, f"字段 {field_name} 不能为空字符串"
        try:
            value = json.loads(raw)
        except Exception:
            return None, f"字段 {field_name} 必须是数值数组或数组 JSON 字符串"

    if isinstance(value, (int, float, np.number)):
        return None, f"字段 {field_name} 必须是数组，不能是单个数值"

    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32, copy=False)
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(list(value), dtype=np.float32)
        except Exception:
            return None, f"字段 {field_name} 无法转换为有效浮点数组"
    else:
        return None, f"字段 {field_name} 类型非法，必须是数组"

    if arr.ndim == 0:
        return None, f"字段 {field_name} 必须是数组，不能是标量"

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        return None, f"字段 {field_name} 不能为空"

    return arr.astype(np.float32), None


def parse_iq_input(data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    支持两种输入方式：
    1. iq: [[I...], [Q...]]
    2. i_points + q_points

    关键修复：
    - 避免对 0 维 numpy 数组直接 len(...)，从而触发 len() of unsized object
    - 允许字符串形式的 JSON 数组
    - 当输入不合法时返回可读错误，由上层自动回退 rule，不再直接 500
    """
    if "iq" in data:
        iq = data.get("iq")

        if isinstance(iq, str):
            raw = iq.strip()
            if not raw:
                return None, "字段 iq 不能为空字符串"
            try:
                iq = json.loads(raw)
            except Exception:
                return None, "字段 iq 必须是长度为 2 的二维数组，例如 [[I...],[Q...]]"

        if not isinstance(iq, (list, tuple)) or len(iq) != 2:
            return None, "字段 iq 必须是长度为 2 的二维数组，例如 [[I...],[Q...]]"

        i_arr, i_err = parse_numeric_sequence(iq[0], "iq[0]")
        if i_err:
            return None, i_err

        q_arr, q_err = parse_numeric_sequence(iq[1], "iq[1]")
        if q_err:
            return None, q_err

        if i_arr.shape[0] != q_arr.shape[0]:
            return None, "I/Q 序列长度不一致"

        return np.stack([i_arr, q_arr], axis=0).astype(np.float32), None

    if "i_points" in data and "q_points" in data:
        i_arr, i_err = parse_numeric_sequence(data.get("i_points"), "i_points")
        if i_err:
            return None, i_err

        q_arr, q_err = parse_numeric_sequence(data.get("q_points"), "q_points")
        if q_err:
            return None, q_err

        if i_arr.shape[0] != q_arr.shape[0]:
            return None, "字段 i_points / q_points 长度不一致"

        return np.stack([i_arr, q_arr], axis=0).astype(np.float32), None

    return None, "当前请求未提供 iq 或 i_points/q_points，无法执行 CNN 推理"

def load_cnn_model_if_needed() -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    global _CNN_MODEL, _CNN_MODEL_META

    if _CNN_MODEL is not None and _CNN_MODEL_META["loaded"]:
        return _CNN_MODEL, _CNN_MODEL_META

    checkpoint_path = Path(MODEL_CONFIG["cnn_checkpoint_path"])
    if not checkpoint_path.exists():
        _CNN_MODEL_META["loaded"] = False
        _CNN_MODEL_META["error"] = f"未找到模型文件：{checkpoint_path}"
        print(f"[WARN] {_CNN_MODEL_META['error']}")
        return None, _CNN_MODEL_META

    try:
        device = get_compute_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)

        label_map = checkpoint.get("label_map", MODEL_CONFIG["label_map"])
        idx_to_name = checkpoint.get(
            "idx_to_name",
            {int(v): k for k, v in label_map.items()}
        )

        model = OneDCNNClassifier(
            num_classes=int(checkpoint.get("num_classes", MODEL_CONFIG["cnn_num_classes"]))
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        _CNN_MODEL = model
        _CNN_MODEL_META = {
            "loaded": True,
            "checkpoint_path": str(checkpoint_path),
            "model_name": checkpoint.get("model_name", MODEL_CONFIG["cnn_model_name"]),
            "input_shape": checkpoint.get("input_shape", [2, MODEL_CONFIG["cnn_input_length"]]),
            "label_map": label_map,
            "idx_to_name": {int(k): v for k, v in idx_to_name.items()} if isinstance(idx_to_name, dict) else idx_to_name,
            "error": None,
            "device": str(device),
        }

        print(f"[INFO] CNN 模型加载成功：{checkpoint_path}")
        return _CNN_MODEL, _CNN_MODEL_META
    except Exception as e:
        _CNN_MODEL = None
        _CNN_MODEL_META["loaded"] = False
        _CNN_MODEL_META["error"] = f"模型加载失败：{e}"
        print(f"[WARN] {_CNN_MODEL_META['error']}")
        return None, _CNN_MODEL_META


def determine_risk_level(peak_power_dbm: float, snr_db: float, power_threshold: float, snr_threshold: float) -> str:
    should_alarm = (peak_power_dbm >= power_threshold) or (snr_db <= snr_threshold)
    if peak_power_dbm >= power_threshold + 5 or snr_db <= snr_threshold - 3:
        return "HIGH"
    if should_alarm:
        return "MEDIUM"
    return "LOW"


def predict_cnn(data: Dict[str, Any], threshold_map: Dict[str, float], threshold_source: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    model, meta = load_cnn_model_if_needed()
    if model is None:
        return None, meta.get("error") or "CNN 模型不可用"

    iq, iq_error = parse_iq_input(data)
    if iq is None:
        return None, iq_error

    target_len = int(meta.get("input_shape", [2, MODEL_CONFIG["cnn_input_length"]])[1])
    iq = resize_iq_to_target_length(iq, target_len)

    if MODEL_CONFIG.get("cnn_normalize_per_channel", True):
        iq = normalize_iq_per_channel(iq)

    x = torch.tensor(iq, dtype=torch.float32).unsqueeze(0).to(get_compute_device())

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    idx_to_name = meta.get("idx_to_name", {v: k for k, v in MODEL_CONFIG["label_map"].items()})
    predicted_label = idx_to_name.get(pred_idx, f"class_{pred_idx}")

    peak_power_dbm = float(data.get("peak_power_dbm", -90))
    snr_db = float(data.get("snr_db", 0))

    power_threshold = float(threshold_map["alarm.power.threshold.dbm"])
    snr_threshold = float(threshold_map["alarm.snr.threshold.db"])
    should_alarm = (peak_power_dbm >= power_threshold) or (snr_db <= snr_threshold)
    risk_level = determine_risk_level(peak_power_dbm, snr_db, power_threshold, snr_threshold)

    reason = (
        f"已使用 1D-CNN 模型完成推理，预测标签={predicted_label}，"
        f"置信度={confidence:.4f}。"
    )

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "should_alarm": should_alarm,
        "reason": reason,
        "model_name": meta.get("model_name", MODEL_CONFIG["cnn_model_name"]),
        "inference_mode": "cnn",
        "actual_mode": "cnn",
        "fallback_used": False,
        "fallback_reason": "",
        "thresholds": {
            "power_alarm_threshold_dbm": power_threshold,
            "snr_alarm_threshold_db": snr_threshold,
            "source": threshold_source
        }
    }, None


# =========================
# 推理模式控制
# =========================
def resolve_predict_mode(data: Dict[str, Any]) -> str:
    mode = str(data.get("model_type", MODEL_CONFIG["default_mode"])).strip().lower()
    if mode not in ("rule", "cnn", "auto"):
        mode = MODEL_CONFIG["default_mode"]
    return mode


def predict_with_fallback(data: Dict[str, Any], threshold_map: Dict[str, float], threshold_source: str) -> Dict[str, Any]:
    mode = resolve_predict_mode(data)

    if mode == "rule":
        result = predict_rule(data, threshold_map, threshold_source)
        result["request_mode"] = mode
        result["actual_mode"] = "rule"
        return result

    if mode in ("cnn", "auto"):
        try:
            cnn_result, cnn_error = predict_cnn(data, threshold_map, threshold_source)
        except Exception as e:
            cnn_result = None
            cnn_error = f"CNN 推理异常：{e}"

        if cnn_result is not None:
            cnn_result["request_mode"] = mode
            cnn_result["actual_mode"] = "cnn"
            return cnn_result

        if not MODEL_CONFIG.get("allow_rule_fallback", True):
            raise RuntimeError(f"CNN 推理失败，且未启用规则兜底：{cnn_error}")

        fallback_result = predict_rule(data, threshold_map, threshold_source)
        fallback_result["fallback_used"] = True
        fallback_result["request_mode"] = mode
        fallback_result["actual_mode"] = "rule"
        fallback_result["fallback_reason"] = str(cnn_error)
        fallback_result["reason"] = f"CNN 推理未启用，已回退规则模型。原因：{cnn_error} " + fallback_result["reason"]
        return fallback_result

    result = predict_rule(data, threshold_map, threshold_source)
    result["request_mode"] = mode
    result["actual_mode"] = "rule"
    return result



def build_health_payload() -> Dict[str, Any]:
    threshold_map, threshold_source = load_thresholds()
    _, cnn_meta = load_cnn_model_if_needed()

    checkpoint_path = Path(cnn_meta.get("checkpoint_path") or MODEL_CONFIG["cnn_checkpoint_path"]).resolve()
    checkpoint_exists = checkpoint_path.exists()
    checkpoint_is_file = checkpoint_path.is_file()

    default_mode = str(MODEL_CONFIG["default_mode"]).strip().lower()
    allow_rule_fallback = bool(MODEL_CONFIG.get("allow_rule_fallback", True))
    cnn_available = bool(cnn_meta.get("loaded", False))

    if default_mode == "rule":
        risk_flag = False
        risk_level = "LOW"
        risk_reason = "当前默认模式为 RULE，在线推理固定走规则模型。"
    elif cnn_available:
        risk_flag = False
        risk_level = "LOW"
        risk_reason = "CNN 模型已成功加载，AUTO 或 CNN 模式均可正常使用。"
    else:
        risk_flag = True
        risk_level = "HIGH"
        if allow_rule_fallback:
            risk_reason = "默认模式为 AUTO/CNN，但 CNN 当前不可用；后续推理会自动回退到 RULE。"
        else:
            risk_reason = "默认模式为 AUTO/CNN，但 CNN 当前不可用；且未开启 RULE 回退，推理可能直接失败。"

    summary = (
        f"status=UP, defaultMode={default_mode}, "
        f"cnnAvailable={'true' if cnn_available else 'false'}, "
        f"checkpointExists={'true' if checkpoint_exists else 'false'}, "
        f"fallbackRisk={'true' if risk_flag else 'false'}"
    )

    return {
        "service": "radio-spectrum-ai",
        "status": "UP",
        "default_mode": default_mode,
        "allow_rule_fallback": allow_rule_fallback,
        "rule_model": {
            "name": MODEL_CONFIG["rule_model_name"],
            "available": True
        },
        "cnn_model": {
            "name": cnn_meta.get("model_name", MODEL_CONFIG["cnn_model_name"]),
            "available": cnn_available,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_is_file": checkpoint_is_file,
            "device": cnn_meta.get("device"),
            "input_shape": cnn_meta.get("input_shape"),
            "error": cnn_meta.get("error")
        },
        "fallback_risk": {
            "has_risk": risk_flag,
            "level": risk_level,
            "reason": risk_reason
        },
        "thresholds": {
            "power_alarm_threshold_dbm": threshold_map["alarm.power.threshold.dbm"],
            "snr_alarm_threshold_db": threshold_map["alarm.snr.threshold.db"],
            "source": threshold_source
        },
        "summary": summary
    }


# =========================
# 路由
# =========================
@app.get("/health")
def health():
    return success(build_health_payload(), "AI服务正常")


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return fail(400, "请求体不能为空，且必须是 JSON")

        required_fields = [
            "center_freq_mhz",
            "bandwidth_khz",
            "peak_power_dbm",
            "snr_db",
            "power_points"
        ]
        for field in required_fields:
            if field not in data:
                return fail(400, f"缺少必填字段：{field}")

        threshold_map, threshold_source = load_thresholds()
        result = predict_with_fallback(data, threshold_map, threshold_source)
        return success(result, "推理成功")
    except Exception as e:
        app.logger.exception("AI /predict 处理异常")
        return fail(500, f"AI推理异常：{str(e)}")

if __name__ == "__main__":
    app.run(
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        debug=APP_CONFIG["debug"]
    )