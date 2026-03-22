
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from config import APP_CONFIG, MODEL_CONFIG

app = Flask(__name__)


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


def normalize_points(points: Any) -> List[float]:
    """
    把 power_points 统一转成 float 列表
    """
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
    """
    用最小 20% 点估计底噪
    """
    if not points:
        return -90.0

    sorted_points = sorted(points)
    take_n = max(1, int(len(points) * 0.2))
    return round(sum(sorted_points[:take_n]) / take_n, 2)


def estimate_active_width(points: List[float], threshold: float) -> int:
    """
    估计活跃频点数量
    """
    return sum(1 for x in points if x >= threshold)


def predict_rule(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    规则假模型：
    根据频段、占用带宽、曲线宽度、SNR、峰值功率做一个最小可运行分类
    """
    center_freq = float(data.get("center_freq_mhz", 0))
    bandwidth_khz = float(data.get("bandwidth_khz", 0))
    peak_power_dbm = float(data.get("peak_power_dbm", -90))
    snr_db = float(data.get("snr_db", 0))
    occupied_bw = float(data.get("occupied_bandwidth_khz", 0))
    channel_model = str(data.get("channel_model", "UNKNOWN"))
    power_points = normalize_points(data.get("power_points", []))

    if not power_points:
        return {
            "predicted_label": "UNKNOWN",
            "confidence": 0.50,
            "risk_level": "MEDIUM",
            "should_alarm": False,
            "reason": "未提供有效频谱采样点，无法做稳定识别。",
            "model_name": MODEL_CONFIG["model_name"]
        }

    noise_floor = estimate_noise_floor(power_points)
    dynamic_peak = round(max(power_points), 2)
    active_threshold = noise_floor + 3.0
    active_bins = estimate_active_width(power_points, active_threshold)

    if occupied_bw <= 0 and bandwidth_khz > 0 and len(power_points) > 0:
        occupied_bw = round(active_bins * (bandwidth_khz / len(power_points)), 2)


    # 1低频模拟段，更偏 AM / FM
    # 2高频数字段，更偏 BPSK / QPSK / 16QAM
    # 3占用带宽越宽，越可能是 FM / 16QAM / QPSK

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

    # 信道影响轻微修正置信度
    if channel_model in ("Rayleigh", "CarrierOffset", "SampleRateError"):
        confidence = max(0.60, round(confidence - 0.06, 2))
    else:
        confidence = round(confidence, 2)

    power_threshold = MODEL_CONFIG["power_alarm_threshold_dbm"]
    snr_threshold = MODEL_CONFIG["snr_alarm_threshold_db"]

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
        "model_name": MODEL_CONFIG["model_name"]
    }

@app.get("/health")
def health():
    return success({
        "service": "radio-spectrum-ai",
        "model": MODEL_CONFIG["model_name"],
        "status": "UP"
    }, "AI服务正常")

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

        result = predict_rule(data)
        return success(result, "推理成功")
    except Exception as e:
        return fail(500, f"AI推理异常：{str(e)}")


if __name__ == "__main__":
    app.run(
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        debug=APP_CONFIG["debug"]
    )