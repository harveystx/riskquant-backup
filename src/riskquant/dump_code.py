# dump_code.py

from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    print(root)
    src = root

    # src = root / "portfolio" / "payloads"
    src = root / "portfolio" / "blocks"

    out_file = root / "all_code.txt"

    exclude_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        "venv",
    }

    with out_file.open("w", encoding="utf-8") as out:
        for path in sorted(src.rglob("*.py")):
            if any(part in exclude_dirs for part in path.parts):
                continue

            out.write(f"\n\n# ===== FILE: {path.relative_to(root)} =====\n\n")
            out.write(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
