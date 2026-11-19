import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "scripts" / "coc_validation_stage"
sys.path.insert(0, str(STAGE_DIR))

from common import (  # noqa: E402
    SAVED_SEGMENT_PROVENANCE_KEY,
    dedupe_result_rows_by_key,
    discover_coc_yaml_files,
    finalize_json_from_jsonl,
    is_completed_result_row,
    load_coc_autolabeler_records,
    load_done_keys,
    normalize_record,
    parse_coc_yaml,
    resolve_row_video_mode,
    resolve_segment_video_path,
    resolve_video_path,
)


def test_parse_real_style_autolabeler_yaml(tmp_path: Path) -> None:
    """Parse a realistic autolabeler YAML into normalized record fields."""
    clip_dir = tmp_path / "1f47bf7f-d233-480c-b166-7512d8e9ac97"
    clip_dir.mkdir()
    yaml_path = clip_dir / "cot_6300000.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "event_start_frame": 63,
                "event_start_timestamp": 6300000,
                "final_content": {
                    "ego_behavior_schema": {
                        "effect_on_ego_behavior": "Resume speed while keeping lane.",
                    }
                },
                "prompt": [
                    {
                        "role": "user",
                        "content": "[1f47bf7f.camera_front_wide_120fov.mp4, index=130]",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record = parse_coc_yaml(yaml_path)

    assert record is not None
    assert record["clip_id"] == "1f47bf7f-d233-480c-b166-7512d8e9ac97"
    assert record["event_start_timestamp"] == 6300000
    assert record["effect_on_ego_behavior"] == "Resume speed while keeping lane."
    assert record["camera_name"] == "camera_front_wide_120fov"


def test_load_coc_result_dir_and_resolve_segment_video(tmp_path: Path) -> None:
    """Load a result dir and resolve its matching saved segment video."""
    coc_result_dir = tmp_path / "experiments" / "20260609_smoke"
    clip_dir = coc_result_dir / "clip-123"
    clip_dir.mkdir(parents=True)
    (clip_dir / "cot_12600000.yaml").write_text(
        yaml.safe_dump(
            {
                "event_start_timestamp": 12600000,
                "final_content": {
                    "ego_behavior_schema": {
                        "effect_on_ego_behavior": "Keep distance.",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    segment_root = tmp_path / "experiments" / "video_segment"
    segment_file = segment_root / "clip-123" / "clip-123_12600000.mp4"
    segment_file.parent.mkdir(parents=True)
    segment_file.write_bytes(b"mp4")

    records = load_coc_autolabeler_records(coc_result_dir, segment_video_root=segment_root)

    assert len(records) == 1
    assert records[0]["video_path"] == str(segment_file)
    assert discover_coc_yaml_files(coc_result_dir)[0].name == "cot_12600000.yaml"
    assert resolve_segment_video_path("clip-123", 12600000, [segment_root]) == segment_file


def test_resolve_segment_video_path_requires_exact_timestamp_match(tmp_path: Path) -> None:
    """Do not fall back to an arbitrary mp4 when the event timestamp does not match."""
    segment_root = tmp_path / "experiments" / "video_segment"
    clip_dir = segment_root / "clip-123"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip-123_12600000.mp4").write_bytes(b"mp4")

    assert resolve_segment_video_path("clip-123", 12600000, [segment_root]) is not None
    assert resolve_segment_video_path("clip-123", 13900000, [segment_root]) is None


def test_is_completed_result_row_only_for_validated_evaluation() -> None:
    """A row is complete only when it holds a parsed, non-empty evaluation."""
    assert is_completed_result_row(
        {"ok": True, "error": None, "parse_error": None, "evaluation": {"score": 1}}
    )
    # API error -> retryable.
    assert not is_completed_result_row({"ok": False, "error": "api_error:boom", "evaluation": None})
    # Parse error -> retryable.
    assert not is_completed_result_row(
        {"ok": False, "parse_error": "json_parse_failed", "evaluation": None}
    )
    # Score validation error -> retryable.
    assert not is_completed_result_row(
        {
            "ok": False,
            "parse_error": "validation_error:invalid_score:visual_evidence_score",
            "evaluation": None,
        }
    )
    # Missing video -> retryable.
    assert not is_completed_result_row({"ok": False, "error": "missing_video"})
    # Parsed but no expected keys extracted -> retryable.
    assert not is_completed_result_row({"ok": True, "error": None, "evaluation": {}})


def test_load_done_keys_skips_retryable_failures(tmp_path: Path) -> None:
    """Only keys with a validated evaluation are treated as done for --resume."""
    jsonl_path = tmp_path / "stage.jsonl"
    rows = [
        {
            "row_key": "good",
            "ok": True,
            "error": None,
            "parse_error": None,
            "evaluation": {"score": 1},
        },
        {"row_key": "api_fail", "ok": False, "error": "api_error:boom", "evaluation": None},
        {
            "row_key": "parse_fail",
            "ok": False,
            "parse_error": "json_parse_failed",
            "evaluation": None,
        },
        {"row_key": "no_video", "ok": False, "error": "missing_video"},
    ]
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert load_done_keys(jsonl_path) == {"good"}


def test_dedupe_result_rows_prefers_completed_over_failed() -> None:
    """Dedupe keeps the completed row over a retryable failure for a key."""
    failed = {
        "row_key": "clip_1",
        "ok": False,
        "error": "api_error:boom",
        "evaluation": None,
    }
    success = {
        "row_key": "clip_1",
        "ok": True,
        "error": None,
        "parse_error": None,
        "evaluation": {"score": 1},
    }
    assert dedupe_result_rows_by_key([failed, success]) == [success]
    assert dedupe_result_rows_by_key([success, failed]) == [success]


def test_dedupe_result_rows_keeps_latest_when_both_retryable() -> None:
    """Dedupe keeps the latest row when both entries are retryable failures."""
    first = {"row_key": "clip_1", "ok": False, "error": "api_error:first"}
    second = {"row_key": "clip_1", "ok": False, "error": "api_error:second"}
    assert dedupe_result_rows_by_key([first, second]) == [second]


def test_finalize_json_from_jsonl_dedupes_by_row_key(tmp_path: Path) -> None:
    """Finalizing JSONL to JSON dedupes rows by key, preferring completed ones."""
    jsonl_path = tmp_path / "stage.jsonl"
    json_path = tmp_path / "stage.json"
    rows = [
        {"row_key": "a", "ok": False, "error": "api_error:boom", "evaluation": None},
        {
            "row_key": "b",
            "ok": True,
            "error": None,
            "parse_error": None,
            "evaluation": {"score": 2},
        },
        {
            "row_key": "a",
            "ok": True,
            "error": None,
            "parse_error": None,
            "evaluation": {"score": 1},
        },
    ]
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    finalize_json_from_jsonl(jsonl_path, json_path)

    finalized = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["row_key"] for row in finalized] == ["a", "b"]
    assert finalized[0]["evaluation"] == {"score": 1}
    assert finalized[0]["error"] is None


def test_resolve_row_video_mode_per_row_provenance() -> None:
    """Auto picks segment_8s only for tracked saved segments; fallback rows use raw."""
    # Saved-segment hint -> segment_8s.
    assert (
        resolve_row_video_mode("auto", hint_is_saved_segment=True, video_source="hint")
        == "segment_8s"
    )
    # Hint present but not a tracked segment (legacy table) -> raw event alignment.
    assert resolve_row_video_mode("auto", hint_is_saved_segment=False, video_source="hint") == "raw"
    # Fallback to --video-root is always a full raw camera video -> raw.
    assert (
        resolve_row_video_mode("auto", hint_is_saved_segment=True, video_source="video_root")
        == "raw"
    )
    # Explicit override always wins regardless of provenance.
    assert (
        resolve_row_video_mode("segment_8s", hint_is_saved_segment=False, video_source="video_root")
        == "segment_8s"
    )
    assert resolve_row_video_mode("raw", hint_is_saved_segment=True, video_source="hint") == "raw"


def test_resolve_video_path_reports_provenance(tmp_path: Path) -> None:
    """resolve_video_path distinguishes an explicit hint from a --video-root fallback."""
    hint_video = tmp_path / "seg" / "clip-1_100.mp4"
    hint_video.parent.mkdir(parents=True)
    hint_video.write_bytes(b"mp4")

    hint_record = {
        "clip_id": "clip-1",
        "effect_on_ego_behavior": "label",
        "event_start_timestamp": 100,
        "video_path": str(hint_video),
        SAVED_SEGMENT_PROVENANCE_KEY: True,
    }
    row, err = normalize_record(hint_record, 0)
    assert err is None
    assert row is not None
    assert row.hint_is_saved_segment is True
    path, source = resolve_video_path(row, video_root=None, camera_name="cam")
    assert path == hint_video
    assert source == "hint"

    # No hint file; a raw camera video is discovered under --video-root.
    video_root = tmp_path / "raw"
    raw_video = video_root / "clip-2.cam.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"mp4")
    fallback_record = {
        "clip_id": "clip-2",
        "effect_on_ego_behavior": "label",
        "event_start_timestamp": 100,
    }
    fallback_row, err = normalize_record(fallback_record, 1)
    assert err is None
    assert fallback_row is not None
    assert fallback_row.hint_is_saved_segment is False
    path, source = resolve_video_path(fallback_row, video_root=video_root, camera_name="cam")
    assert path == raw_video
    assert source == "video_root"

    # Nothing resolves -> missing.
    path, source = resolve_video_path(fallback_row, video_root=None, camera_name="cam")
    assert path is None
    assert source is None


def test_autolabeler_loader_marks_saved_segment_provenance(tmp_path: Path) -> None:
    """Records backed by a saved 8s segment carry the provenance marker."""
    coc_result_dir = tmp_path / "experiments" / "20260609_smoke"
    clip_dir = coc_result_dir / "clip-123"
    clip_dir.mkdir(parents=True)
    (clip_dir / "cot_12600000.yaml").write_text(
        yaml.safe_dump(
            {
                "event_start_timestamp": 12600000,
                "final_content": {
                    "ego_behavior_schema": {"effect_on_ego_behavior": "Keep distance."}
                },
            }
        ),
        encoding="utf-8",
    )
    segment_root = tmp_path / "experiments" / "video_segment"
    segment_file = segment_root / "clip-123" / "clip-123_12600000.mp4"
    segment_file.parent.mkdir(parents=True)
    segment_file.write_bytes(b"mp4")

    records = load_coc_autolabeler_records(coc_result_dir, segment_video_root=segment_root)
    assert records[0][SAVED_SEGMENT_PROVENANCE_KEY] is True
    row, err = normalize_record(records[0], 0)
    assert err is None
    assert row is not None
    assert row.hint_is_saved_segment is True
