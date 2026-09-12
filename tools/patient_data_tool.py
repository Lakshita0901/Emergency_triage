"""Patient data handling tool.

Merge, normalize, and update patient clinical records.
"""
from typing import Any, Dict


def update_patient_records(
    current_data: Dict[str, Any], new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge and normalize patient record updates.

    Performs a clean dictionary merge preserving nested structures where appropriate.
    """
    merged = dict(current_data)
    for k, v in new_data.items():
        if k in ("vitals", "symptoms") and isinstance(v, dict) and isinstance(merged.get(k), dict):
            nested = dict(merged[k])
            nested.update(v)
            merged[k] = nested
        else:
            merged[k] = v
    return merged
