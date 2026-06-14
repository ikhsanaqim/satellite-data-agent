# Satellite Operations Reference Guide

## Signal Quality Thresholds

### SNR (Signal-to-Noise Ratio)
- Excellent: > 15 dB
- Good: 10–15 dB
- Marginal: 8–10 dB
- Poor: < 8 dB (action required: check antenna alignment, weather conditions)

### RSSI (Received Signal Strength Indicator)
- Strong: > -75 dBm
- Moderate: -75 to -85 dBm
- Weak: -85 to -90 dBm
- Critical: < -90 dBm (terminal may lose connectivity)

### Latency
- Normal (LEO/MEO): 20–150 ms
- Normal (GEO): 500–700 ms
- Elevated: 700–900 ms (possible congestion or rain fade)
- Critical: > 900 ms (investigate backhaul, routing, or hardware)

### Packet Loss
- Acceptable: < 1%
- Degraded: 1–3%
- Critical: > 3% (impacts real-time telemetry and remote operations)

## Common Anomaly Patterns

### Rain Fade
Simultaneous degradation of SNR and RSSI during precipitation events.
Typically affects Ka-band more than Ku-band. Duration: minutes to hours.
Check local weather data to correlate.

### Antenna Misalignment
Gradual SNR degradation over days/weeks without weather correlation.
Often affects a single terminal while neighbors remain stable.
Recommended action: schedule field technician for antenna re-pointing.

### Backhaul Congestion
High latency with normal SNR/RSSI. Often correlates with peak traffic hours.
Check gateway utilization and bandwidth allocation.

### Terminal Hardware Failure
Sudden drop to non-OK status with very low or zero throughput.
May show RSSI = 0 or SNR = 0. Requires terminal replacement or reboot.

## Satellite Systems Reference

### MODIS (Moderate Resolution Imaging Spectroradiometer)
- Platforms: Terra (EOS AM-1), Aqua (EOS PM-1)
- Key data products: MOD09GA (surface reflectance), MOD11A1 (land surface temperature)
- Spatial resolution: 250m (bands 1-2), 500m (bands 3-7), 1km (bands 8-36)

### Sentinel-2
- Operator: ESA (European Space Agency)
- Revisit time: 5 days at equator
- Spatial resolution: 10m (4 bands), 20m (6 bands), 60m (3 bands)
- Useful for: vegetation monitoring, land use change, water quality

### Landsat 8/9
- Operator: USGS/NASA
- Revisit time: 16 days per satellite (8 days combined)
- Spatial resolution: 30m multispectral, 15m panchromatic
