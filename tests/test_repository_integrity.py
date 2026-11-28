"""Regression tests for the repository integrity checker."""

from pathlib import Path

from tools.check_repository_integrity import (
    check_artifacts,
    check_markdown_links,
    check_target,
    document_anchors,
    link_targets,
    strip_code_and_comments,
)


def write_file(root: Path, name: str, content: str = "") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_relative_markdown_link(tmp_path: Path) -> None:
    target = write_file(tmp_path, "docs/results.md", "# Results\n")
    source = write_file(
        tmp_path,
        "README.md",
        "[Results](docs/results.md)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert target.exists()
    assert checked == 1
    assert errors == []


def test_missing_relative_markdown_link(tmp_path: Path) -> None:
    source = write_file(
        tmp_path,
        "README.md",
        "[Missing report](docs/missing.md)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 1
    assert len(errors) == 1
    assert "missing local target" in errors[0]
    assert "docs/missing.md" in errors[0]


def test_external_urls_are_not_checked(tmp_path: Path) -> None:
    source = write_file(
        tmp_path,
        "README.md",
        "[GitHub](https://github.com/example/project)\n[Email](mailto:example@example.com)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 2
    assert errors == []


def test_valid_heading_anchor(tmp_path: Path) -> None:
    target = write_file(
        tmp_path,
        "docs/results.md",
        "# Research Results\n\n## Held-out Evaluation\n",
    )
    source = write_file(
        tmp_path,
        "README.md",
        "[Evaluation](docs/results.md#held-out-evaluation)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 1
    assert errors == []
    assert "held-out-evaluation" in document_anchors(target)


def test_missing_heading_anchor(tmp_path: Path) -> None:
    write_file(tmp_path, "docs/results.md", "# Results\n")
    source = write_file(
        tmp_path,
        "README.md",
        "[Wrong anchor](docs/results.md#missing-heading)\n",
    )

    _, errors = check_markdown_links(tmp_path, [source])

    assert len(errors) == 1
    assert "missing Markdown anchor" in errors[0]


def test_same_document_heading_anchor(tmp_path: Path) -> None:
    source = write_file(
        tmp_path,
        "README.md",
        "# Setup and Reproduction\n\n[Jump](#setup-and-reproduction)\n",
    )

    _, errors = check_markdown_links(tmp_path, [source])

    assert errors == []


def test_image_link(tmp_path: Path) -> None:
    write_file(tmp_path, "reports/plot.png", "placeholder")
    source = write_file(
        tmp_path,
        "README.md",
        "![Results figure](reports/plot.png)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 1
    assert errors == []


def test_reference_style_links(tmp_path: Path) -> None:
    write_file(tmp_path, "docs/study.md", "# Study\n")
    source = write_file(
        tmp_path,
        "README.md",
        "[Study][research]\n\n[research]: docs/study.md\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked >= 1
    assert errors == []


def test_code_fences_are_ignored(tmp_path: Path) -> None:
    source = write_file(
        tmp_path,
        "README.md",
        "```markdown\n[Example](missing.md)\n```\n[Real link](https://example.com)\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 1
    assert errors == []


def test_inline_code_and_comments_are_ignored(tmp_path: Path) -> None:
    source = write_file(
        tmp_path,
        "README.md",
        "`[Example](missing.md)`\n<!-- [Hidden](missing.md) -->\n",
    )

    checked, errors = check_markdown_links(tmp_path, [source])

    assert checked == 0
    assert errors == []


def test_missing_required_artifact(tmp_path: Path) -> None:
    write_file(tmp_path, "README.md", "# Example\n")

    errors = check_artifacts(
        tmp_path,
        required=("README.md", "docs/missing.md"),
    )

    assert len(errors) == 1
    assert "docs/missing.md" in errors[0]


def test_existing_required_artifacts(tmp_path: Path) -> None:
    write_file(tmp_path, "README.md", "# Example\n")
    write_file(tmp_path, "docs/results.md", "# Results\n")

    errors = check_artifacts(
        tmp_path,
        required=("README.md", "docs/results.md"),
    )

    assert errors == []


def test_repository_path_escape_rejected(tmp_path: Path) -> None:
    source = write_file(tmp_path, "docs/README.md", "# Example\n")

    problem = check_target(
        tmp_path,
        source,
        "../../../outside.md",
        {},
    )

    assert problem is not None
    assert "escapes repository" in problem


def test_duplicate_heading_anchors(tmp_path: Path) -> None:
    target = write_file(
        tmp_path,
        "README.md",
        "# Results\n\n## Results\n\n## Results\n",
    )

    anchors = document_anchors(target)

    assert "results" in anchors
    assert "results-1" in anchors
    assert "results-2" in anchors


def test_link_extraction_line_numbers() -> None:
    content = "# Example\n\n[Report](docs/report.md)\n![Chart](figures/chart.png)\n"

    targets = link_targets(content)

    assert (3, "docs/report.md") in targets
    assert (4, "figures/chart.png") in targets


def test_strip_code_preserves_real_links() -> None:
    content = "```python\n[Fake](missing.md)\n```\n[Real](README.md)\n"

    cleaned = strip_code_and_comments(content)

    assert "[Fake](missing.md)" not in cleaned
    assert "[Real](README.md)" in cleaned
