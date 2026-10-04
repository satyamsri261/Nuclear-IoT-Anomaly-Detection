"""Monitoring-only energy/process status manager.

Unlike the mentor reference, this version does not switch or control loads.
It only classifies simulated telemetry for dashboard display. This keeps the
prototype focused on detection rather than operational control.
"""

from config import CURRENT_NORMAL, POWER_FACTOR_NORMAL, VOLTAGE_NORMAL


def classify_power_state(data):
    voltage = float(data.get("voltage", 0))
    current = float(data.get("current", 0))
    pf = float(data.get("power_factor", 0))

    if not VOLTAGE_NORMAL[0] <= voltage <= VOLTAGE_NORMAL[1]:
        return "VOLTAGE FLUCTUATION"
    if not CURRENT_NORMAL[0] <= current <= CURRENT_NORMAL[1]:
        return "CURRENT FLUCTUATION"
    if not POWER_FACTOR_NORMAL[0] <= pf <= POWER_FACTOR_NORMAL[1]:
        return "POWER FACTOR FLUCTUATION"
    return "NORMAL OPERATION"
