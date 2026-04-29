from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 9300,
    "debug": True
}

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "mygo",
    "password": "123456",
    "database": "radio_spectrum_monitor",
    "charset": "utf8mb4"
}

MODEL_CONFIG = {
    # rule -> 强制规则模型
    # cnn  -> 优先 CNN，失败后按 allow_rule_fallback 决定是否回退
    # auto -> 有效 IQ + 成功加载模型时走 CNN，否则走 rule
    "default_mode": "auto",

    # 规则模型名称
    "rule_model_name": "rule-model-v1",

    # CNN 模型名称
    "cnn_model_name": "1dcnn-v1",

    # 允许 CNN 失败后自动回退到规则模型
    "allow_rule_fallback": True,

    # 训练输出的模型路径
    "cnn_checkpoint_path": str((BASE_DIR.parent / "ai-research" / "models" / "best_1dcnn.pt").resolve()),

    # CNN 输入长度
    "cnn_input_length": 256,

    # CNN 分类数
    "cnn_num_classes": 5,

    # 每通道标准化
    "cnn_normalize_per_channel": True,

    # 标签兜底映射
    "label_map": {
        "AM": 0,
        "FM": 1,
        "BPSK": 2,
        "QPSK": 3,
        "16QAM": 4
    }
}

THRESHOLD_DEFAULTS = {
    "alarm.power.threshold.dbm": -30.0,
    "alarm.snr.threshold.db": 10.0
}