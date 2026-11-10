"""Score frozen references and package a reviewable local audit report."""

import argparse
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from nba_shots.common import sha256, write_json
from nba_shots.release_audit import (
    checked_references,
    compare_rows,
    read_csv,
    score_audit,
    select_sample,
    summarize,
    verify_freeze,
)


def package(audit, output):
    freeze = verify_freeze(audit)
    refs = checked_references(audit)
    automatic = json.loads((audit / "private/automatic.json").read_text())
    identities = json.loads((audit / "unblinded/identity_reviews.json").read_text())
    pool = read_csv(audit / "private/pool.csv")
    selection = read_csv(audit / "private/selection.csv")
    assert selection == select_sample(pool, freeze["sample_size"], freeze["seed"])
    assert len(pool) == freeze["population"] == len({r["shot_id"] for r in pool})
    assert set(read_csv(audit / "blind/manifest.csv")[0]) == {"shot_id", "clip_path"}
    for row in automatic:
        selected = next(r for r in selection if r["shot_id"] == row["shot_id"])
        if row.get("clip_sha256"):
            assert sha256(selected["clip_path"]) == row["clip_sha256"]
        assert sha256(row["label_path"]) == row["label_sha256"]
    result = score_audit(audit, output)
    comparisons = compare_rows(refs, automatic, identities)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics == summarize(comparisons)
    assert len(read_csv(output / "comparisons.csv")) == len(refs)
    assert sum(metrics["timing_counts"].values()) == len(refs)
    assert (
        sum(metrics["timing_counts"][s] for s in ["within_3", "early", "late", "missing"])
        == metrics["reference_scorable"]
    )
    copies = {
        "freeze.json": "freeze.json",
        "reference_freeze.json": "reference_freeze.json",
        "private/pool.csv": "eligible_pool.csv",
        "private/selection.csv": "frozen_sample.csv",
        "private/automatic.json": "automatic_predictions.json",
        "blind/manifest.csv": "blind_manifest.csv",
        "blind/references.json": "reference_annotations.json",
        "blind/targets.json": "official_target_cues.json",
        "blind/review_protocol.json": "review_protocol.json",
        "blind/alignment.json": "alignment.json",
        "blind/windows.jsonl": "review_windows.jsonl",
        "unblinded/identity_reviews.json": "identity_reviews.json",
    }
    for source, destination in copies.items():
        shutil.copyfile(audit / source, output / destination)
    shutil.copyfile(
        "data/release_audit_source/pool_audit.json", output / "population_provenance.json"
    )
    shutil.copyfile("docs/release_audit.md", output / "methodology.md")
    narrative = """## Method and scope

The pool contained **851 unique video-associated shot events** across 11 games.
Exactly 50 unique IDs were sampled without replacement after sorting IDs, using
`random.Random(42).sample`. Sampling did not use outcomes, detection success, or
feature quality. The original detector from `feature/basketevent-h1` commit
`8a45ea090e5b1bfaae50ceca1460076a5e81305c` was preserved unchanged.

BasketEvent was pinned to `de20e7ec434d6cb0e79e75bdb4de47da84e57c0f`.
Of 1,941 video-associated label files, 851 contained unique shot events. Cached
official shot-chart records linked 850 events unambiguously; one ambiguous event
was retained with a source-derived ID and was not sampled. Live PBP access failed.
All 50 sampled IDs have official matches. This discrepancy was documented before
freezing the sample. See [population provenance](population_provenance.json).

The assistant reviewed uniform contact sheets and visually chosen exact-frame
windows. The reference is the first zero-based decoded frame where the ball has
separated from the shooting hand. Official target-player/period/clock cues located
the intended attempt in multi-attempt clips. Automatic frames, boxes, track
labels, and supplied outcome/success fields stayed hidden during reference review.
Original broadcast scorebugs and visible shot results remained in the decoded
video frames; outcome labels were absent from the blind manifest.
No applicable pre-existing blind annotations were found; older five-clip pilots
were separate cohorts. All 50 references were frozen before unblinding.

Six releases were uncertain and one target release was not visible; none were
replaced. All videos decoded at 25 FPS with trajectory counts equal to decoded
frame counts. Overlays at the reference/identity frames were visually aligned;
there was no observed gross offset. Timing uncertainty and player uncertainty
were recorded separately. A clear identity at an uncertain release uses its saved
`identity_frame`, so role denominators can exceed the timing denominator.

The original automatic track choices were projected onto the reference frame.
Actor agreement was judged visually against the frozen shooter and nearest
opponent references. Defender proximity means image-space feet distance, which
can differ from the primary contesting player. Boxes are approximate supporting
evidence. References were never changed after unblinding.

These are **single-assistant visual references**, without human adjudication or
inter-reviewer agreement. Close feet distances, occlusion, compression, target
identification, and one-frame ambiguity can affect these references. Confidence
intervals describe binomial sampling uncertainty and do not include reference
annotation error or game clustering. Results are limited to this video-available
subset, whose source files use BasketEvent's `test` split. No downstream predictor
was trained or evaluated here. See [reproduction instructions](methodology.md).

## What this means for H1

The small median timing error hides severe failures: three scorable shots were
**82, 111, and 161 frames late** (3.28, 4.44, and 6.44 seconds). The latter two
automatic frames occur during subsequent transition play. The unchanged detector
searches backward through the whole clip for its last qualifying ball-rise event;
these examples motivate constraining the intended shot event before tuning the
release threshold. This is a proposed follow-up, not a detector change in this audit.

Player selection is a separate weakness: shooter agreement was **43/49 (87.8%)**;
defender agreement was **30/46 (65.2%)**. Two selected shooters were different
visible actors. Four other shooter selections were unavailable due to ambiguous
source shooter/event metadata. Defender disagreements include 11 different-actor
selections and five unavailable selections. Some defender differences arise
because tracks were selected at the automatic frame and compared at the reference
frame; this measures the original end-to-end choice, not a corrected choice.

The three missing detections among timing-scorable shots comprise two ambiguous
shooter/event cases and one release-not-found case. Do not summarize this as only
the accuracy among successful detections. Confirmed timing agreement across all
50 selections is **27/50 (54%)**, with **43/50 (86%)** reference timing coverage.

Report these measurements as evidence about the measurement pipeline in H1.
Establishing predictive H1 still requires its planned controlled model comparison.
Any subsequent detector tuning requires a **new blind sample** for an independent
accuracy claim. Reusing these references is suitable for development only.

"""
    report_path = output / "report.md"
    report_path.write_text(
        report_path.read_text().replace("## Cases for review", narrative + "## Cases for review")
    )
    evidence = output / "evidence"
    evidence.mkdir()
    lines = [
        "",
        "## Evidence for every sampled shot",
        "",
        "Red boxes show the selected automatic shooter; yellow boxes show the selected "
        "automatic defender. Cyan boxes are approximate frozen reference boxes. Actor "
        "agreement was judged visually, separately from timing. All images use the "
        "reference frame, or the recorded identity frame when timing was unscorable.",
        "",
        "| Shot | Timing | Error (frames) | Shooter | Defender | Evidence |",
        "|---|---|---:|---|---|---|",
    ]
    for row in comparisons:
        shot = row["shot_id"].replace(":", "_")
        source = audit / "unblinded" / f"{shot}_identity.png"
        Image.open(source).convert("RGB").save(evidence / f"{shot}.jpg", quality=94)
        lines.append(
            f"| {row['shot_id']} | {row['timing_category']} | {row['signed_error']} | "
            f"{row['shooter_match']} | {row['defender_match']} | "
            f"[frame](evidence/{shot}.jpg) |"
        )
    # Fixed representative cases: selected only after references were frozen and scored.
    cases = [
        "nba:0022500020:360",
        "nba:0022500007:718",
        "nba:0022500004:319",
        "nba:0022500007:498",
        "nba:0022500022:30",
    ]
    selection_by_id = {r["shot_id"]: r for r in selection}
    ref_by_id = {r["shot_id"]: r for r in refs}
    auto_by_id = {r["shot_id"]: r for r in automatic}
    gallery = ["", "## Representative timing failures", ""]
    for shot_id in cases:
        ref, auto = ref_by_id[shot_id], auto_by_id[shot_id]
        shot = shot_id.replace(":", "_")
        canvas = Image.new("RGB", (1280, 392), "white")
        source = Path(ref["evidence"][0])
        canvas.paste(Image.open(source).resize((640, 360)), (0, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 8), f"Assistant reference: frame {ref['release_frame']}", fill="black")
        frame = auto["release_frame"]
        if frame is not None:
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    selection_by_id[shot_id]["clip_path"],
                    "-vf",
                    f"select=eq(n\\,{frame})",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            )
            canvas.paste(Image.open(BytesIO(decoded.stdout)).resize((640, 360)), (640, 32))
        draw.text(
            (645, 8), f"Frozen automatic: {frame if frame is not None else 'missing'}", fill="black"
        )
        name = f"{shot}_timing.jpg"
        canvas.save(evidence / name, quality=94)
        gallery.extend(
            [f"### {shot_id}", "", f"![Reference and automatic frame](evidence/{name})", ""]
        )
    with (output / "report.md").open("a") as stream:
        stream.write("\n".join(gallery + lines) + "\n")
    write_json(
        output / "verification.json",
        {
            "deterministic_sample_reproduced": True,
            "unique_population": len(pool),
            "sampled": len(selection),
            "reviewed": len(refs),
            "comparison_rows": len(comparisons),
            "denominators_reconcile": True,
            "frozen_clip_and_label_hashes_match": True,
            "evidence_source_sha256": {
                str(p): sha256(p) for p in [Path(r["evidence"][0]) for r in refs]
            },
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.audit, args.output), indent=2))
