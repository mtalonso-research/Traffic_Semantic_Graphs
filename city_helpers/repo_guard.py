import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(r"F:\NYCU\nuplan_clean\Traffic_Semantic_Graphs")


def digest(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot():
    code = {}
    for pattern in ("*.py", "*.sh", "*.json"):
        for path in REPO.rglob(pattern):
            if ".git" in path.parts or "models" in path.parts or "data" in path.parts:
                continue
            code[str(path.relative_to(REPO))] = digest(path)
    protected = {}
    for folder in (REPO / "models", REPO / "data"):
        for path in folder.rglob("*"):
            if path.is_file():
                stat = path.stat()
                protected[str(path.relative_to(REPO))] = [stat.st_size, stat.st_mtime_ns]
    return {"code_sha256": code, "protected_size_mtime": protected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "verify"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    current = snapshot()
    if args.action == "snapshot":
        if args.manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {args.manifest}")
        args.manifest.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Saved repository guard manifest: {args.manifest}")
        return
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if current != expected:
        raise SystemExit("Repository guard verification FAILED: repository files changed")
    print("Repository guard verification passed: repository unchanged")


if __name__ == "__main__":
    main()
