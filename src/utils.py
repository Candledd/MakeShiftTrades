from datetime import datetime


def log(message: str) -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def parse_timeframe_to_hours(tf):
    if not tf:
        return 0.25
    tf = tf.lower()
    if tf.endswith('m'):
        return float(tf[:-1]) / 60.0
    elif tf.endswith('h'):
        return float(tf[:-1])
    elif tf.endswith('d'):
        return float(tf[:-1]) * 24.0
    else:
        return 0.25
