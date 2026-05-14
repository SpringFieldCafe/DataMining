import csv
import html
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple


CAR_PLATE = "粤BCW7826"
TARGET_DATE = datetime(2025, 9, 25).date()
DATA_FILES = ["BCW7826_2025-09-25.txt", "BCW7826_2025-09-26.txt"]

OUTPUT_TRIPS = "processed_data.csv"
OUTPUT_STATES = "state_timeline.csv"
OUTPUT_GAPS = "recharge_candidates.csv"
OUTPUT_HTML = "BCW7826驾驶行为分析.html"

# The raw files are transaction/trip rows with start/end time and start/end GPS.
# The non-occupied states are inferred from gaps between consecutive trips.
RECHARGE_GAP_MINUTES = 30
EMPTY_DRIVE_SPEED_KMH = 25.0


@dataclass
class Trip:
    row_id: int
    source_file: str
    upload_first: str
    upload_last: str
    duplicate_count: int
    plate: str
    start_time: datetime
    end_time: datetime
    duration_min: float
    fare_or_meter: float
    distance_km: float
    extra_metric: float
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float


@dataclass
class StateSegment:
    state: str
    start_time: datetime
    end_time: datetime
    duration_min: float
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    source: str
    evidence: str
    candidate_rank: int = 0


@dataclass
class RechargeCandidate:
    rank: int
    score: float
    gap_start: datetime
    gap_end: datetime
    gap_min: float
    gap_distance_km: float
    estimated_drive_min: float
    estimated_stationary_min: float
    charger_lon: float
    charger_lat: float
    last_trip_end: datetime
    last_trip_end_lon: float
    last_trip_end_lat: float
    next_trip_start: datetime
    next_trip_start_lon: float
    next_trip_start_lat: float
    last_trip_row: int
    next_trip_row: int
    evidence: str


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def dt_from_ms(value: str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def interpolate_point(
    lon1: float, lat1: float, lon2: float, lat2: float, ratio: float
) -> Tuple[float, float]:
    ratio = max(0.0, min(1.0, ratio))
    return lon1 + (lon2 - lon1) * ratio, lat1 + (lat2 - lat1) * ratio


def load_unique_trips() -> List[Trip]:
    grouped: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for file_name in DATA_FILES:
        path = os.path.join(script_dir(), file_name)
        with open(path, "r", encoding="utf-8", newline="") as f:
            for raw_row in csv.reader(f):
                if len(raw_row) < 17:
                    continue

                # Rows with identical trip time, distance and coordinates are repeated uploads.
                # Keep one trip and record how many duplicate uploads existed as evidence.
                key = tuple(raw_row[3:17])
                if key not in grouped:
                    grouped[key] = {
                        "row": raw_row,
                        "source_file": file_name,
                        "uploads": [],
                    }
                grouped[key]["uploads"].append(raw_row[0])

    trips: List[Trip] = []
    for row_id, item in enumerate(grouped.values(), start=1):
        row = item["row"]
        uploads = sorted(item["uploads"])
        start_time = dt_from_ms(row[3])
        end_time = dt_from_ms(row[4])
        if start_time.date() != TARGET_DATE:
            continue
        trips.append(
            Trip(
                row_id=row_id,
                source_file=item["source_file"],
                upload_first=uploads[0],
                upload_last=uploads[-1],
                duplicate_count=len(uploads),
                plate=CAR_PLATE,
                start_time=start_time,
                end_time=end_time,
                duration_min=(end_time - start_time).total_seconds() / 60,
                fare_or_meter=float(row[5]),
                distance_km=float(row[7]),
                extra_metric=float(row[8]),
                start_lon=float(row[13]),
                start_lat=float(row[14]),
                end_lon=float(row[15]),
                end_lat=float(row[16]),
            )
        )

    trips.sort(key=lambda trip: trip.start_time)
    for row_id, trip in enumerate(trips, start=1):
        trip.row_id = row_id
    return trips


def estimate_recharge_candidates(trips: List[Trip]) -> List[RechargeCandidate]:
    candidates: List[RechargeCandidate] = []

    for previous_trip, next_trip in zip(trips, trips[1:]):
        gap_min = (next_trip.start_time - previous_trip.end_time).total_seconds() / 60
        if gap_min < RECHARGE_GAP_MINUTES:
            continue

        gap_distance_km = haversine_km(
            previous_trip.end_lon,
            previous_trip.end_lat,
            next_trip.start_lon,
            next_trip.start_lat,
        )
        estimated_drive_min = min(
            gap_min * 0.7,
            (gap_distance_km / EMPTY_DRIVE_SPEED_KMH) * 60 if gap_distance_km else 0,
        )
        estimated_stationary_min = gap_min - estimated_drive_min

        # Long stationary time is the strongest signal for recharging. A shorter
        # displacement between adjacent occupied trips also increases confidence.
        score = estimated_stationary_min + min(30, gap_min / 4) - min(25, gap_distance_km * 1.5)
        if estimated_stationary_min < 20:
            continue

        charger_lon, charger_lat = interpolate_point(
            previous_trip.end_lon,
            previous_trip.end_lat,
            next_trip.start_lon,
            next_trip.start_lat,
            0.5,
        )
        evidence = (
            f"上一单 {previous_trip.end_time:%H:%M} 在 "
            f"({previous_trip.end_lon:.6f}, {previous_trip.end_lat:.6f}) 结束；"
            f"下一单 {next_trip.start_time:%H:%M} 从 "
            f"({next_trip.start_lon:.6f}, {next_trip.start_lat:.6f}) 开始；"
            f"中间空档 {gap_min:.1f} 分钟，端点直线距离 {gap_distance_km:.2f} km，"
            f"估计静止 {estimated_stationary_min:.1f} 分钟。"
        )
        candidates.append(
            RechargeCandidate(
                rank=0,
                score=score,
                gap_start=previous_trip.end_time,
                gap_end=next_trip.start_time,
                gap_min=gap_min,
                gap_distance_km=gap_distance_km,
                estimated_drive_min=estimated_drive_min,
                estimated_stationary_min=estimated_stationary_min,
                charger_lon=charger_lon,
                charger_lat=charger_lat,
                last_trip_end=previous_trip.end_time,
                last_trip_end_lon=previous_trip.end_lon,
                last_trip_end_lat=previous_trip.end_lat,
                next_trip_start=next_trip.start_time,
                next_trip_start_lon=next_trip.start_lon,
                next_trip_start_lat=next_trip.start_lat,
                last_trip_row=previous_trip.row_id,
                next_trip_row=next_trip.row_id,
                evidence=evidence,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


def build_state_timeline(trips: List[Trip], candidates: List[RechargeCandidate]) -> List[StateSegment]:
    candidate_by_gap = {
        (candidate.last_trip_row, candidate.next_trip_row): candidate for candidate in candidates
    }
    segments: List[StateSegment] = []

    for trip in trips:
        segments.append(
            StateSegment(
                state="occupied",
                start_time=trip.start_time,
                end_time=trip.end_time,
                duration_min=trip.duration_min,
                start_lon=trip.start_lon,
                start_lat=trip.start_lat,
                end_lon=trip.end_lon,
                end_lat=trip.end_lat,
                source=f"trip row {trip.row_id}",
                evidence=f"交易/行程记录：{trip.start_time:%H:%M:%S}-{trip.end_time:%H:%M:%S}，里程 {trip.distance_km:.3f} km。",
            )
        )

    for previous_trip, next_trip in zip(trips, trips[1:]):
        gap_min = (next_trip.start_time - previous_trip.end_time).total_seconds() / 60
        if gap_min <= 0:
            continue

        candidate = candidate_by_gap.get((previous_trip.row_id, next_trip.row_id))
        gap_distance_km = haversine_km(
            previous_trip.end_lon,
            previous_trip.end_lat,
            next_trip.start_lon,
            next_trip.start_lat,
        )

        if candidate:
            drive_total = candidate.estimated_drive_min
            heading_min = max(1.0, min(gap_min * 0.25, drive_total / 2))
            cruising_min = max(1.0, min(gap_min * 0.25, drive_total - heading_min))
            recharge_min = max(0.0, gap_min - heading_min - cruising_min)
            charger_lon = candidate.charger_lon
            charger_lat = candidate.charger_lat

            heading_start = previous_trip.end_time
            heading_end = heading_start + timedelta(minutes=heading_min)
            recharge_start = heading_end
            recharge_end = recharge_start + timedelta(minutes=recharge_min)
            cruising_start = recharge_end
            cruising_end = next_trip.start_time

            segments.extend(
                [
                    StateSegment(
                        state="heading",
                        start_time=heading_start,
                        end_time=heading_end,
                        duration_min=heading_min,
                        start_lon=previous_trip.end_lon,
                        start_lat=previous_trip.end_lat,
                        end_lon=charger_lon,
                        end_lat=charger_lat,
                        source=f"gap row {previous_trip.row_id}->{next_trip.row_id}",
                        evidence="空档前段：从上一单终点向候选充电点移动。",
                        candidate_rank=candidate.rank,
                    ),
                    StateSegment(
                        state="recharging",
                        start_time=recharge_start,
                        end_time=recharge_end,
                        duration_min=recharge_min,
                        start_lon=charger_lon,
                        start_lat=charger_lat,
                        end_lon=charger_lon,
                        end_lat=charger_lat,
                        source=f"gap row {previous_trip.row_id}->{next_trip.row_id}",
                        evidence=f"长时间无交易且估计静止 {candidate.estimated_stationary_min:.1f} 分钟。",
                        candidate_rank=candidate.rank,
                    ),
                    StateSegment(
                        state="cruising",
                        start_time=cruising_start,
                        end_time=cruising_end,
                        duration_min=cruising_min,
                        start_lon=charger_lon,
                        start_lat=charger_lat,
                        end_lon=next_trip.start_lon,
                        end_lat=next_trip.start_lat,
                        source=f"gap row {previous_trip.row_id}->{next_trip.row_id}",
                        evidence="充电结束后到下一单上车点之间的空驶/巡游。",
                        candidate_rank=candidate.rank,
                    ),
                ]
            )
        else:
            segments.append(
                StateSegment(
                    state="cruising",
                    start_time=previous_trip.end_time,
                    end_time=next_trip.start_time,
                    duration_min=gap_min,
                    start_lon=previous_trip.end_lon,
                    start_lat=previous_trip.end_lat,
                    end_lon=next_trip.start_lon,
                    end_lat=next_trip.start_lat,
                    source=f"gap row {previous_trip.row_id}->{next_trip.row_id}",
                    evidence=f"两单间空档 {gap_min:.1f} 分钟，端点直线距离 {gap_distance_km:.2f} km，按空驶/巡游处理。",
                )
            )

    segments.sort(key=lambda item: (item.start_time, item.end_time, item.state))
    return segments


def write_csv(path: str, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_dt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def save_outputs(trips: List[Trip], segments: List[StateSegment], candidates: List[RechargeCandidate]) -> None:
    out_dir = script_dir()

    trip_rows = []
    for trip in trips:
        row = asdict(trip)
        row["start_time"] = serialize_dt(trip.start_time)
        row["end_time"] = serialize_dt(trip.end_time)
        trip_rows.append(row)
    write_csv(
        os.path.join(out_dir, OUTPUT_TRIPS),
        trip_rows,
        [
            "row_id",
            "source_file",
            "upload_first",
            "upload_last",
            "duplicate_count",
            "plate",
            "start_time",
            "end_time",
            "duration_min",
            "fare_or_meter",
            "distance_km",
            "extra_metric",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
        ],
    )

    segment_rows = []
    for segment in segments:
        row = asdict(segment)
        row["start_time"] = serialize_dt(segment.start_time)
        row["end_time"] = serialize_dt(segment.end_time)
        segment_rows.append(row)
    write_csv(
        os.path.join(out_dir, OUTPUT_STATES),
        segment_rows,
        [
            "state",
            "start_time",
            "end_time",
            "duration_min",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
            "source",
            "evidence",
            "candidate_rank",
        ],
    )

    candidate_rows = []
    for candidate in candidates:
        row = asdict(candidate)
        for key, value in list(row.items()):
            if isinstance(value, datetime):
                row[key] = serialize_dt(value)
        candidate_rows.append(row)
    write_csv(
        os.path.join(out_dir, OUTPUT_GAPS),
        candidate_rows,
        [
            "rank",
            "score",
            "gap_start",
            "gap_end",
            "gap_min",
            "gap_distance_km",
            "estimated_drive_min",
            "estimated_stationary_min",
            "charger_lon",
            "charger_lat",
            "last_trip_end",
            "last_trip_end_lon",
            "last_trip_end_lat",
            "next_trip_start",
            "next_trip_start_lon",
            "next_trip_start_lat",
            "last_trip_row",
            "next_trip_row",
            "evidence",
        ],
    )


def status_summary(segments: List[StateSegment]) -> Dict[str, float]:
    result = defaultdict(float)
    for segment in segments:
        result[segment.state] += segment.duration_min
    return dict(sorted(result.items()))


def build_html(trips: List[Trip], segments: List[StateSegment], candidates: List[RechargeCandidate]) -> str:
    all_lons = [x.start_lon for x in segments] + [x.end_lon for x in segments]
    all_lats = [x.start_lat for x in segments] + [x.end_lat for x in segments]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    pad_lon = max((max_lon - min_lon) * 0.08, 0.005)
    pad_lat = max((max_lat - min_lat) * 0.08, 0.005)
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat

    def point(lon: float, lat: float) -> Tuple[float, float]:
        x = (lon - min_lon) / (max_lon - min_lon) * 1000
        y = 680 - (lat - min_lat) / (max_lat - min_lat) * 680
        return x, y

    colors = {
        "occupied": "#d94b42",
        "heading": "#f0a51a",
        "recharging": "#2e9d64",
        "cruising": "#2f7fd1",
    }
    labels = {
        "occupied": "载客 occupied",
        "heading": "找充电站 heading",
        "recharging": "充电 recharging",
        "cruising": "巡游/空驶 cruising",
    }

    def info_attrs(title: str, detail: str) -> str:
        return (
            f'data-info-title="{html.escape(title, quote=True)}" '
            f'data-info-detail="{html.escape(detail, quote=True)}" tabindex="0"'
        )

    svg_lines = []
    for segment in segments:
        x1, y1 = point(segment.start_lon, segment.start_lat)
        x2, y2 = point(segment.end_lon, segment.end_lat)
        width = 1.2 if segment.state == "occupied" else 2.8
        opacity = 0.35 if segment.state == "occupied" else 0.88
        dash = "6 4" if segment.state in {"heading", "cruising"} else ""
        title = (
            f"{labels[segment.state]} | {segment.start_time:%m-%d %H:%M}"
            f" - {segment.end_time:%H:%M}"
        )
        detail = (
            f"时长 {segment.duration_min:.1f} 分钟；"
            f"起点 ({segment.start_lon:.6f}, {segment.start_lat:.6f})，"
            f"终点 ({segment.end_lon:.6f}, {segment.end_lat:.6f})；"
            f"{segment.evidence}"
        )
        svg_lines.append(
            f'<line class="map-line state-{segment.state} candidate-{segment.candidate_rank}" '
            f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{colors[segment.state]}" stroke-width="{width}" opacity="{opacity}" '
            f'stroke-dasharray="{dash}"></line>'
            f'<line class="hit-line state-{segment.state} candidate-{segment.candidate_rank}" '
            f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'{info_attrs(title, detail)}></line>'
        )

    svg_points = []
    for candidate in candidates[:10]:
        x, y = point(candidate.charger_lon, candidate.charger_lat)
        title = f"候选充电点 #{candidate.rank} | {candidate.gap_start:%m-%d %H:%M}-{candidate.gap_end:%H:%M}"
        detail = (
            f"坐标 ({candidate.charger_lon:.6f}, {candidate.charger_lat:.6f})；"
            f"空档 {candidate.gap_min:.1f} 分钟；"
            f"端点距离 {candidate.gap_distance_km:.2f} km；"
            f"估计静止 {candidate.estimated_stationary_min:.1f} 分钟；"
            f"{candidate.evidence}"
        )
        svg_points.append(
            f'<g class="charger candidate-{candidate.rank}" {info_attrs(title, detail)}>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{11 if candidate.rank == 1 else 8}" '
            f'fill="{colors["recharging"]}" stroke="#ffffff" stroke-width="2"></circle>'
            f'<text x="{x + 13:.2f}" y="{y + 4:.2f}">#{candidate.rank}</text></g>'
        )

    first = candidates[0] if candidates else None
    complete_chain = []
    if first:
        chain_segments = [
            s for s in segments if s.candidate_rank == first.rank or s.source == f"trip row {first.last_trip_row}"
        ]
        chain_segments = [
            s
            for s in chain_segments
            if s.source == f"trip row {first.last_trip_row}" or s.candidate_rank == first.rank
        ]
        chain_segments.sort(key=lambda s: s.start_time)
        for segment in chain_segments:
            if segment.source == f"trip row {first.last_trip_row}" or segment.candidate_rank == first.rank:
                complete_chain.append(segment)

    summary = status_summary(segments)
    total_minutes = sum(summary.values())
    summary_cards = "".join(
        f'<div class="stat"><span>{labels.get(k, k)}</span><strong>{v:.1f} 分钟</strong>'
        f'<small>{(v / total_minutes * 100 if total_minutes else 0):.1f}%</small></div>'
        for k, v in summary.items()
    )

    candidate_table = "".join(
        "<tr>"
        f"<td>{c.rank}</td>"
        f"<td>{c.gap_start:%m-%d %H:%M} - {c.gap_end:%H:%M}</td>"
        f"<td>{c.gap_min:.1f}</td>"
        f"<td>{c.gap_distance_km:.2f}</td>"
        f"<td>{c.estimated_stationary_min:.1f}</td>"
        f"<td>{c.charger_lon:.6f}, {c.charger_lat:.6f}</td>"
        f"<td>{html.escape(c.evidence)}</td>"
        "</tr>"
        for c in candidates
    )

    chain_html = "".join(
        f'<div class="chain-item {segment.state}"><strong>{labels[segment.state]}</strong>'
        f'<span>{segment.start_time:%Y-%m-%d %H:%M:%S} - {segment.end_time:%H:%M:%S}</span>'
        f'<em>{segment.duration_min:.1f} 分钟</em>'
        f'<p>{html.escape(segment.evidence)}</p></div>'
        for segment in complete_chain
    )

    min_time = min(s.start_time for s in segments)
    max_time = max(s.end_time for s in segments)
    total_seconds = (max_time - min_time).total_seconds()
    timeline_blocks = []
    for segment in segments:
        left = (segment.start_time - min_time).total_seconds() / total_seconds * 100
        width = max((segment.end_time - segment.start_time).total_seconds() / total_seconds * 100, 0.08)
        cls = f"block {segment.state} candidate-{segment.candidate_rank}"
        title = (
            f"{labels[segment.state]} {segment.start_time:%m-%d %H:%M}"
            f"-{segment.end_time:%H:%M} ({segment.duration_min:.1f} 分钟)"
        )
        detail = (
            f"起点 ({segment.start_lon:.6f}, {segment.start_lat:.6f})，"
            f"终点 ({segment.end_lon:.6f}, {segment.end_lat:.6f})；"
            f"{segment.evidence}"
        )
        timeline_blocks.append(
            f'<div class="{cls}" style="left:{left:.4f}%;width:{width:.4f}%;" '
            f'{info_attrs(title, detail)}></div>'
        )

    data_json = json.dumps(
        {
            "candidates": [
                {
                    "rank": c.rank,
                    "gap_start": serialize_dt(c.gap_start),
                    "gap_end": serialize_dt(c.gap_end),
                    "charger": [c.charger_lon, c.charger_lat],
                    "evidence": c.evidence,
                }
                for c in candidates
            ],
            "bounds": [min_lon, min_lat, max_lon, max_lat],
        },
        ensure_ascii=False,
    )

    best_text = ""
    if first:
        best_text = (
            f"最可能的完整状态链：上一单在 {first.last_trip_end:%Y-%m-%d %H:%M:%S} 完成，"
            f"随后约 {first.gap_min:.1f} 分钟没有交易记录；根据两端点距离 {first.gap_distance_km:.2f} km "
            f"和估计静止 {first.estimated_stationary_min:.1f} 分钟，推断车辆经历 "
            f"occupied -> heading -> recharging -> cruising。候选充电点约为 "
            f"({first.charger_lon:.6f}, {first.charger_lat:.6f})。"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{CAR_PLATE} 驾驶行为状态识别</title>
  <style>
    :root {{
      --occupied: {colors["occupied"]};
      --heading: {colors["heading"]};
      --recharging: {colors["recharging"]};
      --cruising: {colors["cruising"]};
      --ink: #1d232b;
      --muted: #64717f;
      --line: #d8dee6;
      --bg: #f7f8fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 36px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; letter-spacing: 0; }}
    p {{ line-height: 1.7; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 22px; }}
    section {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .note {{ margin: 0; color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ border-left: 5px solid #9aa6b2; padding: 10px 12px; background: #fafbfc; }}
    .stat span, .stat small {{ display: block; color: var(--muted); }}
    .stat strong {{ display: block; margin: 6px 0; font-size: 18px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    .map-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }}
    svg {{ width: 100%; min-width: 860px; height: auto; display: block; }}
    .map-line {{ pointer-events: none; }}
    .hit-line {{ stroke: transparent; stroke-width: 14; fill: none; pointer-events: stroke; cursor: pointer; }}
    .hit-line.is-active {{ stroke: rgba(29, 35, 43, 0.14); }}
    .grid-line {{ stroke: #e8edf2; stroke-width: 1; }}
    .axis-label {{ fill: #6e7a87; font-size: 12px; }}
    .charger {{ cursor: pointer; }}
    .charger.is-active circle {{ stroke: #1d232b; stroke-width: 3; }}
    .charger text {{ fill: #1d232b; font-size: 13px; font-weight: 700; pointer-events: none; }}
    .info-panel {{
      margin: 12px 0;
      padding: 12px 14px;
      min-height: 88px;
      border: 1px solid var(--line);
      border-left: 5px solid #9aa6b2;
      border-radius: 6px;
      background: #ffffff;
    }}
    .info-panel strong {{ display: block; margin-bottom: 6px; }}
    .info-panel p {{ margin: 0; color: var(--muted); }}
    .timeline {{
      position: relative;
      height: 48px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: repeating-linear-gradient(90deg, #ffffff 0, #ffffff 4.15%, #f1f4f7 4.15%, #f1f4f7 4.2%);
      overflow: hidden;
    }}
    .block {{ position: absolute; top: 8px; height: 32px; opacity: 0.8; }}
    .block {{ cursor: pointer; }}
    .block.is-active {{ outline: 2px solid #1d232b; outline-offset: -2px; opacity: 1; }}
    .block.occupied {{ background: var(--occupied); }}
    .block.heading {{ background: var(--heading); }}
    .block.recharging {{ background: var(--recharging); }}
    .block.cruising {{ background: var(--cruising); }}
    .chain {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .chain-item {{ border-top: 5px solid #9aa6b2; background: #fafbfc; padding: 12px; border-radius: 6px; min-height: 148px; }}
    .chain-item.occupied {{ border-color: var(--occupied); }}
    .chain-item.heading {{ border-color: var(--heading); }}
    .chain-item.recharging {{ border-color: var(--recharging); }}
    .chain-item.cruising {{ border-color: var(--cruising); }}
    .chain-item strong, .chain-item span, .chain-item em {{ display: block; }}
    .chain-item span, .chain-item em {{ color: var(--muted); margin-top: 7px; font-style: normal; }}
    .chain-item p {{ margin: 10px 0 0; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f5f8; }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 900px) {{
      main {{ padding: 12px; }}
      header {{ padding: 20px 16px; }}
      .stats, .chain {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{CAR_PLATE} 驾驶行为状态识别</h1>
      <p class="note">分析日期：{TARGET_DATE}。数据已按行程起止时间、距离、起终点坐标去重；非载客状态由相邻交易行程之间的时间空档和 GPS 端点推断。</p>
  </header>
  <main>
    <section>
      <h2>结论</h2>
      <p>{html.escape(best_text)}</p>
      <div class="stats">{summary_cards}</div>
      <div class="legend">
        <span><i class="dot" style="background:var(--occupied)"></i>载客 occupied</span>
        <span><i class="dot" style="background:var(--heading)"></i>找充电站 heading</span>
        <span><i class="dot" style="background:var(--recharging)"></i>充电 recharging</span>
        <span><i class="dot" style="background:var(--cruising)"></i>巡游/空驶 cruising</span>
      </div>
    </section>

    <section>
      <h2>最可能的一组 occupied -> heading -> recharging -> cruising</h2>
      <div class="chain">{chain_html}</div>
    </section>

    <section>
      <h2>点位可视化</h2>
      <p class="note">红线为载客交易行程；绿色圆点是候选充电点，编号越小越可能。把鼠标放到线段、圆点或时间色块上，下方证据面板会更新。</p>
      <div id="infoPanel" class="info-panel" aria-live="polite">
        <strong>当前证据</strong>
        <p>将鼠标放到图中的线段、绿色充电点，或下面时间线的色块上查看具体时间、坐标和判断依据。</p>
      </div>
      <div class="map-wrap">
        <svg viewBox="0 0 1000 720" role="img" aria-label="GPS endpoints and inferred charging candidates">
          <rect x="0" y="0" width="1000" height="720" fill="#fbfcfd"></rect>
          <g>
            <line class="grid-line" x1="0" y1="136" x2="1000" y2="136"></line>
            <line class="grid-line" x1="0" y1="272" x2="1000" y2="272"></line>
            <line class="grid-line" x1="0" y1="408" x2="1000" y2="408"></line>
            <line class="grid-line" x1="0" y1="544" x2="1000" y2="544"></line>
            <line class="grid-line" x1="200" y1="0" x2="200" y2="680"></line>
            <line class="grid-line" x1="400" y1="0" x2="400" y2="680"></line>
            <line class="grid-line" x1="600" y1="0" x2="600" y2="680"></line>
            <line class="grid-line" x1="800" y1="0" x2="800" y2="680"></line>
          </g>
          <g>{''.join(svg_lines)}</g>
          <g>{''.join(svg_points)}</g>
          <text class="axis-label" x="10" y="705">经度 {min_lon:.4f} - {max_lon:.4f}，纬度 {min_lat:.4f} - {max_lat:.4f}</text>
        </svg>
      </div>
    </section>

    <section>
      <h2>时间线</h2>
      <p class="note">范围：{min_time:%Y-%m-%d %H:%M:%S} 到 {max_time:%Y-%m-%d %H:%M:%S}。悬停色块可查看时间。</p>
      <div class="timeline">{''.join(timeline_blocks)}</div>
    </section>

    <section>
      <h2>候选充电窗口证据</h2>
      <table>
        <thead>
          <tr>
            <th>排名</th>
            <th>空档</th>
            <th>空档分钟</th>
            <th>端点距离 km</th>
            <th>估计静止分钟</th>
            <th>候选点位</th>
            <th>证据</th>
          </tr>
        </thead>
        <tbody>{candidate_table}</tbody>
      </table>
    </section>

    <section>
      <h2>如何复查</h2>
      <p>本报告旁边会生成三个 CSV：<code>{OUTPUT_TRIPS}</code> 是去重后的载客行程，<code>{OUTPUT_STATES}</code> 是完整状态时间线，<code>{OUTPUT_GAPS}</code> 是候选充电窗口及证据。候选充电点不是原始表中直接给出的充电桩坐标，而是由上一单终点、下一单起点和无交易空档推断出的可能点位。</p>
    </section>
  </main>
  <script type="application/json" id="analysis-data">{html.escape(data_json)}</script>
  <script>
    const infoPanel = document.getElementById('infoPanel');
    let activeInfoElement = null;

    function showInfo(element) {{
      if (!element || !infoPanel) return;
      if (activeInfoElement) activeInfoElement.classList.remove('is-active');
      activeInfoElement = element;
      activeInfoElement.classList.add('is-active');
      const title = element.dataset.infoTitle || '证据';
      const detail = element.dataset.infoDetail || '';
      infoPanel.innerHTML = '<strong>' + title + '</strong><p>' + detail + '</p>';
    }}

    document.querySelectorAll('[data-info-title]').forEach(function(element) {{
      element.addEventListener('mouseenter', function() {{ showInfo(element); }});
      element.addEventListener('focus', function() {{ showInfo(element); }});
      element.addEventListener('click', function() {{ showInfo(element); }});
    }});
  </script>
</body>
</html>"""


def create_html_visualization(
    trips: List[Trip], segments: List[StateSegment], candidates: List[RechargeCandidate]
) -> None:
    html_text = build_html(trips, segments, candidates)
    with open(os.path.join(script_dir(), OUTPUT_HTML), "w", encoding="utf-8") as f:
        f.write(html_text)


def print_summary(trips: List[Trip], segments: List[StateSegment], candidates: List[RechargeCandidate]) -> None:
    print(f"分析日期: {TARGET_DATE}")
    print(f"读取并去重后的载客行程数: {len(trips)}")
    duplicate_uploads = sum(trip.duplicate_count - 1 for trip in trips)
    print(f"删除/合并的重复上传记录数: {duplicate_uploads}")
    print(f"状态段数: {len(segments)}")
    print("状态时长:")
    for state, minutes in status_summary(segments).items():
        print(f"  {state}: {minutes:.1f} 分钟")

    if candidates:
        best = candidates[0]
        print("\n最可能的完整状态链:")
        print(
            f"  occupied: 最后一单 {best.last_trip_end:%Y-%m-%d %H:%M:%S} 完成，"
            f"终点 ({best.last_trip_end_lon:.6f}, {best.last_trip_end_lat:.6f})"
        )
        print(
            f"  heading/recharging/cruising: {best.gap_start:%Y-%m-%d %H:%M:%S}"
            f" 到 {best.gap_end:%H:%M:%S} 的空档，候选点"
            f" ({best.charger_lon:.6f}, {best.charger_lat:.6f})"
        )
        print(
            f"  证据: 空档 {best.gap_min:.1f} 分钟，端点距离 {best.gap_distance_km:.2f} km，"
            f"估计静止 {best.estimated_stationary_min:.1f} 分钟。"
        )
    else:
        print("未找到达到阈值的充电候选窗口。")

    print("\n输出文件:")
    for name in [OUTPUT_HTML, OUTPUT_TRIPS, OUTPUT_STATES, OUTPUT_GAPS]:
        print(f"  {os.path.join(script_dir(), name)}")


def main() -> None:
    trips = load_unique_trips()
    candidates = estimate_recharge_candidates(trips)
    segments = build_state_timeline(trips, candidates)
    save_outputs(trips, segments, candidates)
    create_html_visualization(trips, segments, candidates)
    print_summary(trips, segments, candidates)


if __name__ == "__main__":
    main()
