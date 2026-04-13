import os
from pathlib import Path


def generate_template_tree(directory, prefix="", align_col=50):
    """
    Recursively generates a tree string, padding every line to a specific
    column and adding a '#' so the user can easily fill in descriptions.
    """
    # Hide common clutter folders
    ignore_dirs = {
        ".git",
        ".github",
        ".pytest_cache",
        "__pycache__",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".venv",
        "env",
        ".env" ".idea",
        ".vscode",
        "pyproject.toml",
        "README.md",
        ".idea",
    }
    ignore_files = {
        ".DS_Store",
        "_generate_project_structure.py",
        "_create_master_context.py",
        ".gitignore",
        "master_context.txt",
        "project_structure.txt",
    }

    path = Path(directory)
    if not path.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    tree_str = ""
    paths = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    paths = [
        p
        for p in paths
        if p.name not in (ignore_dirs if p.is_dir() else ignore_files)
    ]

    for i, p in enumerate(paths):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "

        # Add a trailing slash for directories
        display_name = f"{p.name}/" if p.is_dir() else p.name

        # Build the structural part of the line
        tree_part = f"{prefix}{connector}{display_name}"

        # Calculate padding to make the '#' align perfectly
        padding_length = max(2, align_col - len(tree_part))
        padding = " " * padding_length

        # Add the aligned comment placeholder
        tree_str += f"{tree_part}{padding} \n"

        # Recurse into directories
        if p.is_dir():
            extension = "    " if is_last else "│   "
            tree_str += generate_template_tree(p, prefix + extension, align_col)

    return tree_str


def save_tree_to_file(directory, output_filename="project_structure.txt"):
    # Added .resolve() to get the true folder name even if directory is '.'
    target_path = Path(directory).resolve()

    # Updated Header format
    header = f"Directory Structure for: {target_path.name}\n"
    header += "=" * len(f"Directory Structure for: {target_path.name}") + "\n\n"
    header += f"{target_path.name}/\n"

    # Generate the tree with aligned placeholders (adjust 50 if your filenames are very long)
    tree_output = generate_template_tree(target_path, align_col=50)
    full_output = header + tree_output

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_output)

    print("\n===============================================================")
    print(f"Success! Project structure saved to '{output_filename}'")
    print("===============================================================\n")


if __name__ == "__main__":
    # Point this to your target directory ('.' means the current folder)
    target_folder = "."
    save_tree_to_file(target_folder)
