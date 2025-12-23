import time # Added this import
from datetime import datetime
from zoneinfo import ZoneInfo # Added for timezone handling

def _get_current_time_stage() -> str:
    """Determines the current time stage ('high' or 'low') based on UTC+8 time."""
    # Define the timezone for China Standard Time (UTC+8)
    tz_shanghai = ZoneInfo("Asia/Shanghai")
    
    # Get the current time in the specified timezone
    now = datetime.now(tz_shanghai)
    
    is_weekend = now.weekday() >= 5  # 5 for Saturday, 6 for Sunday
    is_high_hours = now.hour >= 0 and now.hour <= 7  # 0 AM to 7 AM
    
    if is_weekend or is_high_hours:
        return 'extra_high'
    else:
        return 'high'
    
print(f"Current time stage: {_get_current_time_stage()}")