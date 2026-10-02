from pathlib import Path


def generate_template_tree(
    directory, prefix="", align_col=50, include_top_level=None, is_top_level=False
):
    """
    Recursively generates a tree string, padding every line to a specific
    column and adding a ' ' so the user can easily fill in descriptions.
    """
    # Hide common clutter folders
    ignore_dirs = {
        ".git",
        ".github",
        ".pytest_cache",
        "__pycache__",
        ".ruff_cache",
        ".ipynb_checkpoints",
        "venv",
        ".venv",
        ".env",
        "env",
        ".env.idea",
        ".vscode",
        ".idea",
        "riskquant.egg-info",
        "_deprecated",
        "_deprecated_v1",
        "_deprecated_v2",
        ".hypothesis",
    }
    ignore_files = {
        ".DS_Store",
        "_generate_project_structure.py",
        "_create_master_context.py",
        "_get_imported_packages.py",
        ".gitignore",
        "master_context.txt",
        "project_structure.txt",
    }

    path = Path(directory)
    if not path.is_dir():
        return f"Error: '{directory}' is not a valid directory."

    tree_str = ""
    paths = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    # Filter paths
    filtered_paths = []
    for p in paths:
        # 1. Apply standard ignores
        if p.name in (ignore_dirs if p.is_dir() else ignore_files):
            continue

        # 2. Apply top-level filtering (only affects directories at the root level)
        if is_top_level and include_top_level is not None:
            if p.is_dir() and p.name not in include_top_level:
                continue

        filtered_paths.append(p)

    paths = filtered_paths

    for i, p in enumerate(paths):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "

        # Add a trailing slash for directories
        display_name = f"{p.name}/" if p.is_dir() else p.name

        # Build the structural part of the line
        tree_part = f"{prefix}{connector}{display_name}"

        # Calculate padding to make spacing align perfectly
        padding_length = max(2, align_col - len(tree_part))
        padding = " " * padding_length

        # Add the aligned comment placeholder
        tree_str += f"{tree_part}{padding} \n"

        # Recurse into directories (turn off is_top_level for subdirectories)
        if p.is_dir():
            extension = "    " if is_last else "│   "
            tree_str += generate_template_tree(
                p, prefix + extension, align_col, include_top_level, is_top_level=False
            )

    return tree_str


def save_tree_to_file(
    directory, output_filename="project_structure.txt", include_top_level=None
):
    # Added .resolve() to get the true folder name even if directory is '.'
    target_path = Path(directory).resolve()
    print(f"Generating project structure for: {target_path}\n")

    # Updated Header format
    header = f"Directory Structure for: {target_path.name}\n"
    header += "=" * len(f"Directory Structure for: {target_path.name}") + "\n\n"
    header += f"{target_path.name}/\n"

    # Generate the tree with aligned placeholders
    tree_output = generate_template_tree(
        target_path,
        align_col=50,
        include_top_level=include_top_level,
        is_top_level=True,
    )
    full_output = header + tree_output

    output = target_path / "_contexts" / output_filename

    # Ensure the _contexts directory exists
    assert output.parent.exists(), (
        f"Error: The directory '{output.parent}' does not exist. Please create it first."
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(full_output)

    heading_str = f"Success! Project structure saved to '{output}'"

    header_line = "=" * len(heading_str)
    print("\n" + header_line)
    print(heading_str)
    print(header_line + "\n")


if __name__ == "__main__":
    # Point this to your target directory ('.' means the current folder)
    target_folder = "."

    # Define which top-level folders to include. Set to None to include all folders.
    folders_to_include = [
        "docs",
        "scripts",
        "src",
        "tests",
    ]

    save_tree_to_file(target_folder, include_top_level=folders_to_include)
