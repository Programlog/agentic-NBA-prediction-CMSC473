"""Cache video-associated BasketEvent labels and official PBP without revealing predictions."""

import concurrent.futures
import json
import ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

ROOT = Path("data/release_audit_source")
REVISION = "de20e7ec434d6cb0e79e75bdb4de47da84e57c0f"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return True
    for attempt in range(4):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=45,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as r:
                content = r.read()
            json.loads(content)
            destination.with_suffix(destination.suffix + ".part").write_bytes(content)
            destination.with_suffix(destination.suffix + ".part").replace(destination)
            return True
        except Exception:
            time.sleep(attempt + 1)
    return False


def main():
    if not (ROOT / "dataset_info.json").exists():
        if not download(
            f"https://huggingface.co/api/datasets/zaywas/BasketEvent/revision/{REVISION}",
            ROOT / "dataset_info.json",
        ):
            raise RuntimeError("Could not retrieve pinned dataset inventory")
    info = json.loads((ROOT / "dataset_info.json").read_text())
    revision = info["sha"]
    if revision != REVISION:
        raise ValueError("Cached inventory differs from the audit's pinned source revision")
    paths = sorted(
        x["rfilename"][:-4] + ".json" for x in info["siblings"] if x["rfilename"].endswith(".mp4")
    )

    def one(path):
        return download(
            f"https://huggingface.co/datasets/zaywas/BasketEvent/resolve/{revision}/{path}",
            ROOT / "labels" / path,
        )

    done = failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for ok in pool.map(one, paths):
            done += 1
            failures += not ok
            if done % 100 == 0:
                print(json.dumps({"labels_processed": done, "failed": failures}), flush=True)
    games = sorted({path.split("/")[1] for path in paths})
    for game in games:
        ok = download(
            f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game}.json",
            ROOT / "pbp" / f"{game}.json",
        )
        print(json.dumps({"pbp_game": game, "downloaded": ok}), flush=True)
    print(json.dumps({"labels": len(paths), "failed": failures, "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
