"""Generate a blind-safe, auditable checkpoint from human release review CSVs.

This script deliberately does not import or access detector predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

RELEASE_COLUMNS = {
    "video_path",
    "manual_release_frame",
    "manual_release_valid",
    "manual_error_type",
    "reviewer_notes",
    "review_status",
}
IDENTITY_COLUMNS = {
    "video_path",
    "event_status",
    "player_name",
    "team",
    "shot_subtype",
    "description",
    "identity_decision",
    "observed_shooter",
    "observed_action",
    "identity_notes",
}
REVIEW_STATUSES = {"provisional", "unreviewed", "uncertain"}
IDENTITY_DECISIONS = {"pending", "confirmed", "mismatch", "uncertain"}


def read_review_csv(path: Path, required_columns: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required_columns - fields
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = {}
        for line, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
            video_path = row["video_path"]
            if not video_path:
                raise ValueError(f"{path}:{line}: blank video_path")
            if video_path in rows:
                raise ValueError(f"{path}:{line}: duplicate video_path {video_path}")
            rows[video_path] = row
    return rows


def parse_frame_interval(value: str) -> tuple[int, int]:
    parts = value.replace("–", "-").split("-")
    if len(parts) not in {1, 2} or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"invalid manual_release_frame: {value!r}")
    start = int(parts[0].strip())
    end = int(parts[-1].strip())
    if end < start:
        raise ValueError(f"descending manual_release_frame: {value!r}")
    return start, end


def safe_cell(value: str) -> str:
    """Escape markdown table delimiters and line breaks."""
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(release_path: Path, identity_path: Path) -> str:
    release = read_review_csv(release_path, RELEASE_COLUMNS)
    identity = read_review_csv(identity_path, IDENTITY_COLUMNS)
    if set(release) != set(identity):
        only_release = sorted(set(release) - set(identity))
        only_identity = sorted(set(identity) - set(release))
        raise ValueError(
            f"CSV video_path sets differ; release-only={only_release}; "
            f"identity-only={only_identity}"
        )

    for path, row in release.items():
        status = row["review_status"]
        if status not in REVIEW_STATUSES:
            raise ValueError(f"{path}: unexpected review_status {status!r}")
        frame = row["manual_release_frame"]
        if status == "provisional":
            if not frame:
                raise ValueError(f"{path}: provisional entry has no manual_release_frame")
            parse_frame_interval(frame)
        elif frame:
            raise ValueError(f"{path}: {status} entry unexpectedly contains manual_release_frame")
        if status == "uncertain" and not row["reviewer_notes"]:
            raise ValueError(f"{path}: uncertain entry needs reviewer_notes")

    for path, row in identity.items():
        decision = row["identity_decision"]
        if decision not in IDENTITY_DECISIONS:
            raise ValueError(f"{path}: unexpected identity_decision {decision!r}")
        if decision != "pending" and not row["identity_notes"]:
            raise ValueError(f"{path}: {decision} identity decision needs identity_notes")
        if decision == "confirmed" and (not row["observed_shooter"] or not row["observed_action"]):
            raise ValueError(
                f"{path}: confirmed identity needs observed_shooter and observed_action"
            )

    statuses = Counter(row["review_status"] for row in release.values())
    decisions = Counter(row["identity_decision"] for row in identity.values())
    confirmed_provisional = sum(
        release[path]["review_status"] == "provisional"
        and identity[path]["identity_decision"] == "confirmed"
        for path in release
    )
    lines = [
        "# BasketEvent human release-review checkpoint (blind-safe)",
        "",
        "## Scope and limitations",
        "",
        "This report summarizes a **partial manual-review checkpoint**, not a finalized",
        "ground-truth dataset or a detector-accuracy evaluation. The reviewer examined",
        "basketball videos and recorded candidate ball-hand separation intervals.",
        "Some frame interpretations and note wording were assisted by an AI assistant.",
        "Identity decisions marked `confirmed` are the reviewing user's judgments",
        "about matches to the annotated shot events, not independent adjudications.",
        "A `provisional` release interval is **not** a verified release frame.",
        "No detector predictions or timing residuals were accessed for this report.",
        "",
        "## Coverage",
        "",
        f"- Sample clips: **{len(release)}**",
        f"- Provisional release intervals: **{statuses['provisional']}**",
        f"- Unreviewed release clips: **{statuses['unreviewed']}**",
        f"- Uncertain release clips: **{statuses['uncertain']}**",
        f"- User-confirmed event identities: **{decisions['confirmed']}**",
        f"- Pending event identities: **{decisions['pending']}**",
        f"- Uncertain event identities: **{decisions['uncertain']}**",
        f"- Mismatched event identities: **{decisions['mismatch']}**",
        (
            "- Provisional intervals with a user-confirmed event identity: "
            f"**{confirmed_provisional}**"
        ),
        "",
        "## Recorded provisional intervals",
        "",
        (
            "| Video path | Release interval (frames, inclusive) | "
            "Annotated player | Identity decision | Reviewer observation |"
        ),
        "|---|---:|---|---|---|",
    ]
    for path in sorted(release):
        rel = release[path]
        if rel["review_status"] != "provisional":
            continue
        ident = identity[path]
        lines.append(
            "| "
            + " | ".join(
                map(
                    safe_cell,
                    [
                        path,
                        rel["manual_release_frame"],
                        ident["player_name"],
                        ident["identity_decision"],
                        rel["reviewer_notes"],
                    ],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reviewed event-identity judgments",
            "",
            (
                "| Video path | Annotated event | Identity decision | "
                "Observed shooter | Observed action | Review rationale |"
            ),
            "|---|---|---|---|---|---|",
        ]
    )
    for path in sorted(identity):
        ident = identity[path]
        if ident["identity_decision"] == "pending":
            continue
        lines.append(
            "| "
            + " | ".join(
                map(
                    safe_cell,
                    [
                        path,
                        ident["description"],
                        ident["identity_decision"],
                        ident["observed_shooter"],
                        ident["observed_action"],
                        ident["identity_notes"],
                    ],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reproduction and integrity",
            "",
            "Regenerate this Markdown with `python -m tools.report_release_human_review`",
            "using the two local review worksheets. The CSVs may be ignored locally;",
            "therefore this Markdown preserves a readable snapshot, not the original",
            "source worksheets. Keep the originals securely for later adjudication.",
            "",
            f"- Release CSV SHA-256: `{source_sha256(release_path)}`",
            f"- Identity CSV SHA-256: `{source_sha256(identity_path)}`",
            "",
            "**Not evaluated here:** detector accuracy, absolute timing error, inter-rater",
            "agreement, frame-ground-truth validity, or statistical generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--releases", type=Path, default=Path("artifacts/release_manual_review/blind_review.csv")
    )
    parser.add_argument(
        "--identities",
        type=Path,
        default=Path("artifacts/release_manual_review/event_identity_review.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/basketevent_human_release_review.md")
    )
    args = parser.parse_args()
    report = build_report(args.releases, args.identities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
