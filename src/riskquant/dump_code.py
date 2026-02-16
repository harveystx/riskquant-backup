# src\riskquant\dump_code.py

from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    print(root)
    src = root

    src = [
        root / "config",
        root / "portfolio" / "constants.py",
        root / "portfolio" / "payloads",
        root / "portfolio" / "blocks",
        root / "market_data",
    ]
    # src = root / "portfolio"

    out_file = root / "all_code.txt"

    exclude_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        "venv",
    }

    with out_file.open("w", encoding="utf-8") as out:
        for path in sorted(src):
            for subpath in sorted(
                path.rglob("*.py") if path.is_dir() else [path]
            ):
                if any(part in exclude_dirs for part in subpath.parts):
                    continue

                out.write(
                    f"\n\n# ===== FILE: {subpath.relative_to(root)} =====\n\n"
                )
                out.write(subpath.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
