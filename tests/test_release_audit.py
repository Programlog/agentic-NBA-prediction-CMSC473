import json

import pytest

from nba_shots.release_audit import (
    BLIND_COLUMNS,
    compare_rows,
    select_sample,
    summarize,
    wilson_interval,
)


def reference(shot_id="s", frame=10, status="clear", aligned=True):
    return {
        "shot_id": shot_id,
        "release_frame": frame,
        "status": status,
        "alignment_verified": aligned,
        "shooter_status": "clear",
        "defender_status": "clear",
    }


def automatic(shot_id="s", frame=10):
    return {"shot_id": shot_id, "release_frame": frame, "shooter_key": "p0", "defender_key": "p1"}


def identity(shot_id="s", shooter=True, defender=True):
    return {"shot_id": shot_id, "shooter_match": shooter, "defender_match": defender}


def test_sample_is_order_independent_and_blind():
    pool = [
        {
            "shot_id": f"s{i:03}",
            "clip_path": f"{i}.mp4",
            "release_frame": i,
            "outcome": i % 2,
            "automatic_status": "ok",
        }
        for i in range(851)
    ]
    a, b = select_sample(pool), select_sample(list(reversed(pool)))
    assert a == b and len({r["shot_id"] for r in a}) == 50
    assert a != select_sample(pool, seed=43)
    blind = [{k: row[k] for k in BLIND_COLUMNS} for row in a]
    assert all(set(r) == {"shot_id", "clip_path"} for r in blind)
    assert "release_frame" not in json.dumps(blind)
    with pytest.raises(ValueError, match="unique"):
        select_sample([{"shot_id": "same"}, {"shot_id": "same"}], n=1)


@pytest.mark.parametrize(
    "offset,expected", [(-4, "early"), (-3, "within_3"), (3, "within_3"), (4, "late")]
)
def test_inclusive_timing_boundary(offset, expected):
    rows = compare_rows([reference()], [automatic(frame=10 + offset)], [identity()])
    assert rows[0]["timing_category"] == expected


def test_missing_detection_and_unscorable_references_have_distinct_denominators():
    refs = [
        reference("a"),
        reference("b", status="uncertain"),
        reference("c", aligned=False),
        reference("d"),
    ]
    autos = [automatic("a", None), automatic("b"), automatic("c"), automatic("d", 12)]
    roles = [identity(s) for s in ["a", "b", "c", "d"]]
    metrics = summarize(compare_rows(refs, autos, roles))
    assert metrics["sample_size"] == 4
    assert metrics["reference_scorable"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["confirmed_matches_out_of_sample"] == 0.25
    assert metrics["timing_counts"] == {
        "within_3": 1,
        "early": 0,
        "late": 0,
        "missing": 1,
        "unscorable": 2,
    }


def test_identity_mismatch_and_missing_tracks_count_as_failures():
    auto = automatic()
    rows = compare_rows([reference()], [auto], [identity(defender=False)])
    assert summarize(rows)["defender"]["accuracy"] == 0
    auto["shooter_key"] = None
    assert not compare_rows([reference()], [auto], [identity()])[0]["shooter_match"]
    auto["defender_key"] = None
    ref = reference()
    ref["defender_status"] = "none_visible"
    assert compare_rows([ref], [auto], [identity()])[0]["defender_match"]


def test_alignment_failure_excludes_timing_and_identity():
    row = compare_rows(
        [reference(aligned=False)], [automatic()], [identity(shooter=None, defender=None)]
    )[0]
    assert row["timing_category"] == "unscorable"
    assert not row["shooter_scorable"] and not row["defender_scorable"]


def test_comparison_requires_complete_unique_coverage():
    with pytest.raises(ValueError, match="every sampled"):
        compare_rows([reference()], [], [identity()])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_rows([reference(), reference()], [automatic()], [identity()])
    with pytest.raises(ValueError, match="visual"):
        compare_rows([reference()], [automatic()], [identity(shooter=None)])


def test_wilson_and_empty_accuracy():
    lo, hi = wilson_interval(25, 50)
    assert lo == pytest.approx(0.366445, abs=1e-5)
    assert hi == pytest.approx(0.633555, abs=1e-5)
    assert wilson_interval(0, 0) is None
    metrics = summarize(compare_rows([reference(status="unreadable")], [automatic()], [identity()]))
    assert metrics["accuracy"] is None and metrics["wilson_95"] is None


def frozen_fixture(tmp_path):
    from nba_shots.common import sha256, write_json
    from nba_shots.release_audit import write_csv

    (tmp_path / "blind").mkdir(parents=True)
    (tmp_path / "private").mkdir()
    selected = [{"shot_id": "s", "clip_path": "unavailable.mp4"}]
    write_csv(tmp_path / "private/selection.csv", selected, BLIND_COLUMNS)
    write_csv(tmp_path / "blind/manifest.csv", selected, BLIND_COLUMNS)
    write_json(tmp_path / "private/automatic.json", [automatic()])
    ref = reference(status="unreadable", frame=None, aligned=False)
    ref.update(shooter_status="not_visible", defender_status="not_visible", evidence=[])
    write_json(tmp_path / "blind/references.json", [ref])
    write_json(
        tmp_path / "freeze.json",
        {
            "sample_size": 1,
            "automatic_sha256": sha256(tmp_path / "private/automatic.json"),
            "selection_sha256": sha256(tmp_path / "private/selection.csv"),
            "blind_manifest_sha256": sha256(tmp_path / "blind/manifest.csv"),
        },
    )
    return tmp_path


def test_unblind_requires_completed_frozen_references(tmp_path):
    from nba_shots.release_audit import freeze_references, score_audit, unblind_audit

    audit = frozen_fixture(tmp_path / "audit")
    with pytest.raises(FileNotFoundError):
        unblind_audit(audit)
    freeze_references(audit)
    unblind_audit(audit)
    result = score_audit(audit, tmp_path / "report")
    assert result["sample_size"] == 1 and result["accuracy"] is None
    assert "accuracy is undefined" in (tmp_path / "report/report.md").read_text()


def test_frozen_reference_tampering_is_rejected(tmp_path):
    from nba_shots.release_audit import checked_references, freeze_references

    audit = frozen_fixture(tmp_path / "audit")
    freeze_references(audit)
    path = audit / "blind/references.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed after freezing"):
        checked_references(audit)


def test_sampling_stage_does_not_expose_automatic_information(tmp_path, monkeypatch):
    from nba_shots import release_audit as audit

    clip = tmp_path / "opaque.mp4"
    clip.write_bytes(b"synthetic clip")
    pool = tmp_path / "pool.csv"
    rows = [{"shot_id": f"s{i:03}", "clip_path": str(clip), "outcome": i % 2} for i in range(851)]
    audit.write_csv(pool, rows, list(rows[0]))
    monkeypatch.setattr(
        audit,
        "automatic_record",
        lambda row: {
            "shot_id": row["shot_id"],
            "release_frame": 999,
            "shooter_key": "secret_track",
            "defender_key": "other_secret",
            "track_frame_count": 10,
            "automatic_status": "secret_status",
        },
    )
    monkeypatch.setattr(audit, "video_metadata", lambda path: {"frame_count": 10})
    destination = tmp_path / "audit"
    result = audit.sample_audit(pool, destination)
    assert result["population"] == 851 and result["sample_size"] == 50
    blind_text = "".join(p.read_text() for p in (destination / "blind").iterdir())
    assert all(
        secret not in blind_text
        for secret in ["999", "secret_track", "other_secret", "secret_status", "outcome"]
    )
    refs = json.loads((destination / "blind/references.json").read_text())
    assert len(refs) == 50 and all(r["release_frame"] is None for r in refs)
    assert set(audit.read_csv(destination / "blind/manifest.csv")[0]) == set(BLIND_COLUMNS)
