# scripts/run_all_scripts_credit.py
import subprocess
import sys
import time
from pathlib import Path

# Directory containing the scripts you want to run
TARGET_DIR = Path(__file__).parent / "demos" / "credit"


def main():
    # Ensure the target directory exists
    if not TARGET_DIR.exists():
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        sys.exit(1)

    # Get the name of this specific script so it doesn't run itself
    this_script_name = Path(__file__).name

    # Find all Python files recursively in the target directory
    # Excluding __init__.py files AND this runner script
    scripts = [
        p
        for p in TARGET_DIR.rglob("*.py")
        if p.name not in ["__init__.py", this_script_name]
        and not p.name.startswith("_scratch")  # Exclude _scratch scripts
        and not p.name.startswith("audit_")  # Exclude audit scripts
        and not p.name.startswith("run_all_")  # Exclude runner scripts
    ]

    if not scripts:
        print(f"No Python scripts found in {TARGET_DIR}")
        return

    print(f"Found {len(scripts)} scripts to execute:\n")

    # --- NEW CODE ADDED HERE ---
    # Print the list of all found scripts before running them
    for i, script in enumerate(scripts, start=1):
        # Using relative_to makes the output cleaner, showing path from TARGET_DIR
        print(f"  {i:>3d}. {script.relative_to(TARGET_DIR)}")
    print("\n" + "=" * 40 + "\n")
    # ---------------------------

    time.sleep(1)  # Optional: Pause for a moment before starting

    for script in scripts:
        len_script_path = len(str(script))

        # Print the requested heading format
        print(f"\n{'=' * (len_script_path + 20)}")
        print(f"RUNNING: {script.name}")
        print(f"PATH:    {script}")
        print(f"{'=' * (len_script_path + 20)}\n")

        try:
            # sys.executable ensures it uses the same Python interpreter (and venv)
            # running this script
            _ = subprocess.run([sys.executable, str(script)], check=True, text=True)
        except subprocess.CalledProcessError as e:
            # Catches errors if the script exits with a non-zero status code
            print(f"\n[!] ERROR: {script.name} failed with exit code {e.returncode}.")
            print("Stopping execution.")
            sys.exit(e.returncode)

    print("\nAll scripts executed successfully!")


if __name__ == "__main__":
    main()
