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
    "model_name": "rule-model-v1"
}

THRESHOLD_DEFAULTS = {
    "alarm.power.threshold.dbm": -30.0,
    "alarm.snr.threshold.db": 10.0
}