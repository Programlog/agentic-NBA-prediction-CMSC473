"""Read-only checks for repository artifacts and local Markdown links.

Usage:
    python tools/check_repository_integrity.py

External URLs are intentionally not checked. Markdown parsing supports common
inline links, images, reference-style links, and GitHub-style heading anchors.
It is not a complete CommonMark parser.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

REQUIRED_ARTIFACTS = (
    "README.md",
    "docs/h1_results.md",
    "docs/h1_test_protocol.md",
    "docs/proposal_completion_audit.md",
    "docs/final_reproducibility_audit.md",
    "docs/final_research_claim_audit.md",
    "docs/final_demo_readiness.md",
    "notebooks/demo.ipynb",
    "reports/basketevent_h1_test/summary.json",
    "reports/basketevent_h1_test/h1_effect_estimates.png",
    "reports/basketevent_h1_accuracy/reliability.png",
    "experiments/h2/02_final_test_100/README.md",
    "experiments/h2/02_final_test_100/report/combined_metrics.csv",
    "experiments/h2/02_final_test_100/report/figures/h2_1_brier_all_models.png",
    "experiments/h2/02_final_test_100/report/figures/h2_2_effect_intervals.png",
)

INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)")
REFERENCE_DEFINITION = re.compile(
    r"^[ \t]{0,3}\[([^\]\n]+)\]:[ \t]*(<[^>\n]+>|\S+)",
    re.MULTILINE,
)
REFERENCE_LINK = re.compile(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]")
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)\s*#*\s*$", re.MULTILINE)
HTML_ID = re.compile(r'\bid=["\']([^"\']+)["\']')
INLINE_CODE = re.compile(r"(`+)(?!`)(.*?)\1", re.DOTALL)


def strip_code_and_comments(content: str) -> str:
    """Mask code blocks, inline code, and HTML comments while preserving lines."""
    lines = []
    fence_char = None
    fence_length = 0

    for line in content.splitlines(keepends=True):
        match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)

        if fence_char is not None:
            if match and match.group(1)[0] == fence_char:
                if len(match.group(1)) >= fence_length:
                    fence_char = None
                    fence_length = 0

            lines.append("\n" if line.endswith("\n") else "")
            continue

        if match:
            fence_char = match.group(1)[0]
            fence_length = len(match.group(1))
            lines.append("\n" if line.endswith("\n") else "")
            continue

        lines.append(line)

    cleaned = "".join(lines)

    def mask(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    cleaned = re.sub(r"<!--.*?-->", mask, cleaned, flags=re.DOTALL)
    cleaned = INLINE_CODE.sub(mask, cleaned)
    return cleaned


def github_slug(heading: str) -> str:
    """Approximate the GitHub-generated ID for a Markdown heading."""
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = heading.replace("`", "").replace("*", "").replace("_", "")
    heading = heading.strip().lower()

    heading = "".join(
        char
        for char in heading
        if char in "- " or char.isalnum() or unicodedata.category(char).startswith("M")
    )

    return heading.replace(" ", "-")


def document_anchors(path: Path) -> set[str]:
    """Collect Markdown heading anchors and explicitly declared HTML IDs."""
    content = path.read_text(encoding="utf-8-sig")
    without_code = strip_code_and_comments(content)
    anchors = set(HTML_ID.findall(without_code))
    duplicates: dict[str, int] = {}

    for match in HEADING.finditer(without_code):
        slug = github_slug(match.group(1))

        count = duplicates.get(slug, 0)
        duplicates[slug] = count + 1

        anchor = slug if count == 0 else f"{slug}-{count}"
        anchors.add(anchor)

    return anchors


def link_targets(content: str) -> list[tuple[int, str]]:
    """Extract common local Markdown link targets with source line numbers."""
    cleaned = strip_code_and_comments(content)
    result = []

    definitions = {}

    for match in REFERENCE_DEFINITION.finditer(cleaned):
        label = " ".join(match.group(1).lower().split())
        target = match.group(2).strip("<>")
        definitions[label] = target

        line = cleaned.count("\n", 0, match.start()) + 1
        result.append((line, target))

    for match in INLINE_LINK.finditer(cleaned):
        target = match.group(1).strip()

        if target.startswith("<"):
            target = target[1:].split(">", maxsplit=1)[0]
        else:
            # Markdown allows an optional quoted title after the destination.
            target = re.split(r'\s+(?=["\'])', target, maxsplit=1)[0]

        line = cleaned.count("\n", 0, match.start()) + 1
        result.append((line, target))

    for match in REFERENCE_LINK.finditer(cleaned):
        label = match.group(2) or match.group(1)
        label = " ".join(label.lower().split())

        if label in definitions:
            line = cleaned.count("\n", 0, match.start()) + 1
            result.append((line, definitions[label]))

    return result


def check_target(
    root: Path,
    source: Path,
    target: str,
    anchor_cache: dict[Path, set[str]],
) -> str | None:
    """Return an error message for an invalid local target, or None."""
    target = target.strip()

    if not target:
        return None

    if target.startswith(("//", "www.")):
        return None

    parsed = urlsplit(target)

    # URLs, email links, and other URI schemes are outside the offline check.
    if parsed.scheme or parsed.netloc:
        return None

    relative_path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)

    if relative_path.startswith("/"):
        destination = (root / relative_path.lstrip("/")).resolve()
    elif relative_path:
        destination = (source.parent / relative_path).resolve()
    else:
        destination = source.resolve()

    try:
        destination.relative_to(root.resolve())
    except ValueError:
        return f"target escapes repository: {target}"

    if not destination.exists():
        return f"missing local target: {target}"

    if fragment and destination.is_file() and destination.suffix.lower() == ".md":
        if destination not in anchor_cache:
            anchor_cache[destination] = document_anchors(destination)

        if fragment not in anchor_cache[destination]:
            return f"missing Markdown anchor: {target}"

    return None


def check_markdown_links(
    root: Path,
    files: list[Path],
) -> tuple[int, list[str]]:
    """Check links in provided Markdown files and return count and errors."""
    errors = []
    checked = 0
    anchor_cache: dict[Path, set[str]] = {}

    for source in sorted(files):
        if not source.is_file():
            errors.append(f"{source}: Markdown source file does not exist")
            continue

        content = source.read_text(encoding="utf-8-sig")

        for line, target in link_targets(content):
            checked += 1
            problem = check_target(root, source, target, anchor_cache)

            if problem:
                try:
                    display_path = source.relative_to(root)
                except ValueError:
                    display_path = source

                errors.append(f"{display_path}:{line}: {problem}")

    return checked, errors


def check_artifacts(
    root: Path,
    required: tuple[str, ...] = REQUIRED_ARTIFACTS,
) -> list[str]:
    """Check that essential project documentation and outputs exist."""
    errors = []

    for relative_path in required:
        path = root / relative_path

        if not path.is_file():
            errors.append(f"missing required artifact: {relative_path}")

    return errors


def tracked_markdown_files(root: Path) -> list[Path]:
    """Find tracked Markdown files using Git, excluding untracked local data."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        capture_output=True,
        check=True,
    )

    paths = result.stdout.decode("utf-8", errors="replace").split("\0")

    return [root / path for path in paths if path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check local Markdown links and essential repository artifacts."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of tools directory).",
    )

    args = parser.parse_args(argv)
    root = args.root.resolve()

    if not (root / ".git").exists():
        print(f"ERROR: Not a Git repository root: {root}")
        return 2

    print("=== Repository Integrity Check ===")
    print(f"Repository: {root}")

    try:
        markdown_files = tracked_markdown_files(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: Unable to list tracked Markdown files: {exc}")
        return 2

    print(f"Tracked Markdown files: {len(markdown_files)}")
    print(f"Required artifacts: {len(REQUIRED_ARTIFACTS)}")

    artifact_errors = check_artifacts(root)
    link_count, link_errors = check_markdown_links(root, markdown_files)

    print(f"Markdown link references inspected: {link_count}")
    print(f"Missing required artifacts: {len(artifact_errors)}")
    print(f"Invalid local Markdown references: {len(link_errors)}")

    errors = artifact_errors + link_errors

    if errors:
        print("\n=== Findings ===")

        for error in errors:
            print(f"FAIL: {error}")

        print(f"\nFAILED: {len(errors)} finding(s).")
        return 1

    print("\nPASSED: No missing required artifacts or invalid supported local links.")
    print("Note: External URLs and unsupported Markdown syntax were not validated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
