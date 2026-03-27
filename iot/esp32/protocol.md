# ESP32 Telemetry Protocol (MVP)

Endpoint: `POST /api/v1/telemetry`

JSON schema:

```json
{
  "device_id": "esp32-river-01",
  "timestamp_ms": 1710000000000,
  "latitude": 43.2389,
  "longitude": 76.8897,
  "ph": 7.12,
  "turbidity_ntu": 12.8
}
```

Recommended send interval: 1-5 seconds.
Transport: Wi-Fi + HTTPS where available.
