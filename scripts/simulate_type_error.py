#!/usr/bin/env python3
"""
Sensor Data Validation Pipeline - Step 2
Simulates a TypeError caused by mixed data types (strings mixed with floats)
in IoT sensor readings received from an external API.
"""

import sys
import traceback


def compute_statistics(measurements: list) -> dict:
    """
    Compute basic statistics (mean, min, max) for a list of numeric measurements.

    Expects all elements to be int or float.  Will raise TypeError if strings
    are present (e.g. 'N/A', 'ERROR', 'OFFLINE' coming from faulty sensors).
    """
    return {
        "mean": sum(measurements) / len(measurements),
        "min": min(measurements),
        "max": max(measurements),
        "count": len(measurements),
    }


def process_sensor_batch(sensor_records: list) -> None:
    """Iterate over sensor records and compute per-sensor statistics."""
    print(f"[INFO] Processing {len(sensor_records)} sensor record(s)...")

    for idx, record in enumerate(sensor_records, start=1):
        sensor_id = record.get("sensor_id", f"UNKNOWN-{idx}")
        readings = record.get("readings", [])
        unit = record.get("unit", "?")

        print(f"[INFO] Sensor {idx}/{len(sensor_records)}: {sensor_id}")
        print(f"[INFO]   Readings ({len(readings)} values, unit={unit}): {readings}")

        # compute_statistics will blow up because some readings are strings
        stats = compute_statistics(readings)

        print(
            f"[INFO]   mean={stats['mean']:.2f} {unit}, "
            f"min={stats['min']}, max={stats['max']}"
        )
        print()


if __name__ == "__main__":
    print("[INFO] ============================================================")
    print("[INFO]  Step 2: Validate & Process Sensor Readings")
    print("[INFO] ============================================================")
    print("[INFO] Script      : sensor_processor.py")
    print("[INFO] Purpose     : Validate IoT sensor readings received via gateway API")
    print("[INFO] Data source : iot-gateway.internal / topic=sensors/temperature")
    print()

    # Simulated payload from the IoT gateway API.
    # TEMP-001 is clean; TEMP-002 and TEMP-003 have string sentinel values
    # ('N/A', 'ERROR', 'OFFLINE') that the gateway inserts for missing readings.
    sensor_data = [
        {
            "sensor_id": "TEMP-001",
            "readings": [23.4, 24.1, 22.8, 25.0, 23.7],
            "unit": "Celsius",
        },
        {
            "sensor_id": "TEMP-002",
            # Mixed types: floats and string sentinels
            "readings": [21.5, "N/A", 22.3, "ERROR", 21.8, 22.6],
            "unit": "Celsius",
        },
        {
            "sensor_id": "TEMP-003",
            "readings": [19.2, 20.1, "OFFLINE", 19.8],
            "unit": "Celsius",
        },
    ]

    print(f"[INFO] Received {len(sensor_data)} sensor record(s) from IoT gateway.")
    print()

    try:
        process_sensor_batch(sensor_data)
        print("[INFO] All sensor records processed successfully.")
        print("[INFO] Step 2 PASSED.")
    except TypeError as e:
        print()
        print("[ERROR] *** TYPE ERROR ***", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "[ERROR] Step 2 FAILED: non-numeric value encountered in sensor readings.",
            file=sys.stderr,
        )
        print(
            "[ERROR] The IoT gateway is returning string sentinels ('N/A', 'ERROR', "
            "'OFFLINE') instead of skipping or replacing missing readings with NaN.",
            file=sys.stderr,
        )
        print("[ERROR] Traceback (most recent call last):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
