"""Strict JSON formatting for a frozen reducer's Python report.

The reducer represents measured check names as sets. Sorting those names
changes only their serialization; never infer or overwrite pass/admission.
Use outside an immutable experiment source tree to retain its exact hashes.
"""


def json_report(value):
    if isinstance(value, set):
        if any(type(item) is not str for item in value):
            raise TypeError("Report check sets must contain strings only")
        return sorted(value)
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("Report object keys must be strings")
        return {key: json_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_report(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("Unsupported report value: " + type(value).__name__)
