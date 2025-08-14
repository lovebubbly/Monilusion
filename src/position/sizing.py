from datetime import datetime, timedelta

# Global or class variables for circuit breaker
consecutive_losses = 0
is_in_cooldown = False
cooldown_release_time = None

MAX_CONSECUTIVE_LOSSES = 4 # Default value from PDF
COOLDOWN_PERIOD_BARS = 24 # Default value from PDF (1-hour bars, 24 hours)

def get_cooldown_release_time(cooldown_period_bars):
    # Assuming 1 bar = 1 hour for simplicity as per PDF example
    return datetime.now() + timedelta(hours=cooldown_period_bars)

def on_trade_closed(pnl):
    global consecutive_losses, is_in_cooldown, cooldown_release_time
    if pnl <= 0:
        consecutive_losses += 1
    else:
        consecutive_losses = 0

    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        is_in_cooldown = True
        cooldown_release_time = get_cooldown_release_time(COOLDOWN_PERIOD_BARS)

def can_open_new_trade(current_time):
    global is_in_cooldown, consecutive_losses
    if is_in_cooldown:
        if current_time >= cooldown_release_time:
            is_in_cooldown = False
            consecutive_losses = 0 # Reset on cooldown release
        else:
            return False # Still in cooldown
    return True # Not in cooldown or cooldown just released