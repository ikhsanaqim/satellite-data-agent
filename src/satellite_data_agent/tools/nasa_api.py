"""NASA Common Metadata Repository (CMR) API wrapper.

Endpoint: https://cmr.earthdata.nasa.gov/search
No authentication required — this is a public API.

Default region: Indonesian archipelago (bounding_box: 95,-11,141,6)
Default collection: MODIS Terra Surface Reflectance (MOD09GA v6.1)
"""

from __future__ import annotations

import requests
from datetime import datetime, timedelta


CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Indonesian archipelago bounding box: west, south, east, north
DEFAULT_BOUNDING_BOX = "95,-11,141,6"

# MODIS Terra Surface Reflectance Daily
DEFAULT_COLLECTION_CONCEPT_ID = "C2565788712-LPCLOUD"  # MOD09GA v6.1

DEFAULT_PAGE_SIZE = 10
DEFAULT_DAYS_BACK = 7


def fetch_cmr_granules(
    bounding_box: str = DEFAULT_BOUNDING_BOX,
    collection_concept_id: str = DEFAULT_COLLECTION_CONCEPT_ID,
    days_back: int = DEFAULT_DAYS_BACK,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """Fetch recent granule metadata from NASA CMR.

    Args:
        bounding_box: "west,south,east,north" in degrees.
        collection_concept_id: CMR collection concept ID.
        days_back: Number of days to look back from today.
        page_size: Max granules to return.

    Returns:
        dict with keys: records, record_count, source, time_range, sites, devices.
        Each record has: granule_id, dataset, time_start, time_end, bounding_box, links.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "collection_concept_id": collection_concept_id,
        "bounding_box": bounding_box,
        "temporal": f"{start_date.strftime('%Y-%m-%dT00:00:00Z')},{end_date.strftime('%Y-%m-%dT23:59:59Z')}",
        "page_size": page_size,
        "sort_key": "-start_date",
    }

    response = requests.get(CMR_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    entries = data.get("feed", {}).get("entry", [])

    records = []
    for entry in entries:
        record = {
            "granule_id": entry.get("id", ""),
            "title": entry.get("title", ""),
            "dataset": entry.get("dataset_id", ""),
            "time_start": entry.get("time_start", ""),
            "time_end": entry.get("time_end", ""),
            "updated": entry.get("updated", ""),
            "data_center": entry.get("data_center", ""),
            "original_format": entry.get("original_format", ""),
            "links": [
                link.get("href", "")
                for link in entry.get("links", [])
                if link.get("rel", "").endswith("/data#")
            ],
        }

        # Extract bounding box if available
        boxes = entry.get("boxes", [])
        if boxes:
            record["bounding_box"] = boxes[0]

        records.append(record)

    timestamps = [r["time_start"] for r in records if r.get("time_start")]

    return {
        "records": records,
        "record_count": len(records),
        "source": "NASA CMR API",
        "collection_concept_id": collection_concept_id,
        "bounding_box": bounding_box,
        "time_range": {
            "start": min(timestamps) if timestamps else "",
            "end": max(timestamps) if timestamps else "",
        },
        "sites": [],    # CMR doesn't have "sites" — field exists for interface compatibility
        "devices": [],  # same as above
    }
