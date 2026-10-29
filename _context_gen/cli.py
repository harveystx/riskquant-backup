# _context_gen/cli.py
"""
CLI entry point.

One coherent execution model, replacing the previous generator's dead
argparse branch (its `choices`/`default` never matched the real profile
registry, and was unreachable besides -- see the audit trail). Every
`--profile` value this CLI accepts is derived from the live domain
registry at argument-parse time, so it is structurally impossible for the
CLI to advertise a profile name that does not exist.

`tests` regenerates the whole per-domain test split under
`_contexts/tests/` in one pass. There is no per-bucket choice: the split
owns that subdirectory wholesale, so all of its files always come from
one snapshot.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import PurePosixPath

from _context_gen import fs, generate

SPECIAL_PROFILES: tuple[str, ...] = ("all", "index", "core", "tests", "scripts")


def _build_parser(available_domains: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RiskQuant context snapshots."
    )
    parser.add_argument(
        "--profile",
        choices=[*SPECIAL_PROFILES, *available_domains],
        default="all",
        help=(
            "Which artifact(s) to generate. 'all' generates index + core + every "
            "domain + tests + scripts. 'tests' generates the per-domain test split "
            "under _contexts/tests/ (one context_riskquant_tests_<bucket>.txt per "
            "tests/ directory). 'scripts' generates context_riskquant_scripts.txt."
        ),
    )
    parser.add_argument(
        "--tests", action="store_true", help="Include this profile's test source."
    )
    parser.add_argument(
        "--demos",
        action="store_true",
        help="Include this profile's demo/solution scripts.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Publish even if one or more artifacts are INCOMPLETE (each still carries its own status block).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    repo_root = fs.repo_root()
    domains = generate.discover_domains(repo_root) + generate.discover_solution_domains(
        repo_root
    )
    parser = _build_parser(domains)
    args = parser.parse_args(argv)
    domain_names = frozenset(domains)

    snap = generate.take_snapshot(repo_root)
    targets = (
        domains
        if args.profile == "all"
        else [args.profile]
        if args.profile not in SPECIAL_PROFILES
        else []
    )

    rendered: list[generate.RenderedProfile] = []
    for domain in targets:
        profile = generate.build_profile(domain, repo_root)
        rendered.append(
            generate.render_domain_profile(
                profile, snap, repo_root, tests=args.tests, demos=args.demos
            )
        )

    if args.profile in {"all", "core"}:
        rendered.append(generate.render_core_artifact(snap))

    replace_subdirs: frozenset[str] = frozenset()
    if args.profile in {"all", "tests"}:
        rendered.extend(
            generate.render_tests_artifacts(snap, repo_root, domain_names=domain_names)
        )
        replace_subdirs = frozenset({generate.TESTS_SUBDIR})

    if args.profile in {"all", "scripts"}:
        rendered.append(generate.render_scripts_artifact(snap, repo_root))

    index_needed = args.profile in {"all", "index"}
    if index_needed:
        # Index summarises every OTHER artifact this run produced -- if the
        # run is index-only, still summarise the full domain set so the
        # index is never generated in isolation from what it describes.
        summary_source = rendered or [
            *(
                generate.render_domain_profile(
                    generate.build_profile(d, repo_root),
                    snap,
                    repo_root,
                    tests=False,
                    demos=False,
                )
                for d in domains
            ),
            generate.render_core_artifact(snap),
            *generate.render_tests_artifacts(
                snap, repo_root, domain_names=domain_names
            ),
        ]
        rendered.append(generate.render_index_artifact(snap, repo_root, summary_source))

    output_dir = repo_root / "_contexts"
    written = generate.publish(
        rendered,
        output_dir,
        allow_incomplete=args.allow_incomplete,
        replace_subdirs=replace_subdirs,
    )
    tokens_by_name = {rp.filename: rp.token_estimate for rp in rendered}
    for name in written:
        path = output_dir.joinpath(*PurePosixPath(name).parts)
        print(f"WROTE {path}  tokens~={tokens_by_name[name]}")

    print("\n" + "-" * 40)
    print(f"DONE: {datetime.now()}")
    print("-" * 40 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
