from typing import Any, Dict, List, Tuple

import pymysql
from flask import Flask, jsonify, request

from config import APP_CONFIG, DB_CONFIG, MODEL_CONFIG, THRESHOLD_DEFAULTS

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


def get_db_connection():
    """
    创建 MySQL 连接
    """
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
    """
    从 sys_config 读取告警阈值。
    读取失败时，自动退回默认值，保证 AI 服务可用。
    """
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


def predict_rule(data: Dict[str, Any], threshold_map: Dict[str, float], threshold_source: str) -> Dict[str, Any]:
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

    power_threshold = float(threshold_map["alarm.power.threshold.dbm"])
    snr_threshold = float(threshold_map["alarm.snr.threshold.db"])

    if not power_points:
        return {
            "predicted_label": "UNKNOWN",
            "confidence": 0.50,
            "risk_level": "MEDIUM",
            "should_alarm": False,
            "reason": "未提供有效频谱采样点，无法做稳定识别。",
            "model_name": MODEL_CONFIG["model_name"],
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

    # 1 低频模拟段，更偏 AM / FM
    # 2 高频数字段，更偏 BPSK / QPSK / 16QAM
    # 3 占用带宽越宽，越可能是 FM / 16QAM / QPSK
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
        "model_name": MODEL_CONFIG["model_name"],
        "thresholds": {
            "power_alarm_threshold_dbm": power_threshold,
            "snr_alarm_threshold_db": snr_threshold,
            "source": threshold_source
        }
    }


@app.get("/health")
def health():
    threshold_map, threshold_source = load_thresholds()
    return success({
        "service": "radio-spectrum-ai",
        "model": MODEL_CONFIG["model_name"],
        "status": "UP",
        "thresholds": {
            "power_alarm_threshold_dbm": threshold_map["alarm.power.threshold.dbm"],
            "snr_alarm_threshold_db": threshold_map["alarm.snr.threshold.db"],
            "source": threshold_source
        }
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

        threshold_map, threshold_source = load_thresholds()
        result = predict_rule(data, threshold_map, threshold_source)
        return success(result, "推理成功")
    except Exception as e:
        return fail(500, f"AI推理异常：{str(e)}")


if __name__ == "__main__":
    app.run(
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        debug=APP_CONFIG["debug"]
    )