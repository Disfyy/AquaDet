from __future__ import annotations

import argparse
import random
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish mock telemetry to AquaDet API")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--device-id", type=str, default="esp32-river-01")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    with httpx.Client(timeout=5.0) as client:
        for _ in range(args.count):
            payload = {
                "device_id": args.device_id,
                "timestamp_ms": int(time.time() * 1000),
                "latitude": 43.2389 + random.uniform(-0.001, 0.001),
                "longitude": 76.8897 + random.uniform(-0.001, 0.001),
                "ph": round(random.uniform(6.4, 8.6), 2),
                "turbidity_ntu": round(random.uniform(2.0, 70.0), 2),
            }
            resp = client.post(args.url, json=payload)
            resp.raise_for_status()
            print(resp.json())
            time.sleep(0.5)


if __name__ == "__main__":
    main()
