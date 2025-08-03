#!/usr/bin/env python3
"""Release script for mcp-mux."""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def get_current_version() -> str:
    """Get the current version from git tags."""
    result = run_command(["git", "describe", "--tags", "--abbrev=0"], check=False)
    if result.returncode != 0:
        return "0.0.0"
    return result.stdout.strip().lstrip("v")


def validate_version(version: str) -> bool:
    """Validate version format (major.minor.patch)."""
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)


def main():
    """Main release process."""
    parser = argparse.ArgumentParser(description="Release mcp-mux")
    parser.add_argument(
        "version",
        help="Version to release (e.g., 0.2.0)",
        nargs="?",
    )
    parser.add_argument("--major", action="store_true", help="Bump major version")
    parser.add_argument("--minor", action="store_true", help="Bump minor version")
    parser.add_argument("--patch", action="store_true", help="Bump patch version")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    args = parser.parse_args()

    # Determine version
    if args.version:
        version = args.version
    else:
        current = get_current_version()
        major, minor, patch = map(int, current.split("."))

        if args.major:
            version = f"{major + 1}.0.0"
        elif args.minor:
            version = f"{major}.{minor + 1}.0"
        elif args.patch:
            version = f"{major}.{minor}.{patch + 1}"
        else:
            print("Error: Specify version or use --major/--minor/--patch")
            sys.exit(1)

    # Validate version
    if not validate_version(version):
        print(f"Error: Invalid version format: {version}")
        print("Expected format: major.minor.patch (e.g., 0.2.0)")
        sys.exit(1)

    tag = f"v{version}"
    print(f"\nPreparing release {tag}")

    # Check git status
    result = run_command(["git", "status", "--porcelain"])
    if result.stdout.strip():
        print("Error: Working directory is not clean. Commit or stash changes.")
        if not args.dry_run:
            sys.exit(1)

    # Check we're on main branch
    result = run_command(["git", "branch", "--show-current"])
    branch = result.stdout.strip()
    if branch != "main":
        print(f"Warning: Not on main branch (current: {branch})")
        if not args.dry_run:
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                sys.exit(1)

    # Pull latest changes
    if not args.dry_run:
        print("\nPulling latest changes...")
        run_command(["git", "pull", "--rebase"])

    # Run tests
    print("\nRunning tests...")
    if not args.dry_run:
        result = run_command(["uv", "run", "pytest", "tests/"], check=False)
        if result.returncode != 0:
            print("Error: Tests failed!")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)

    # Create and push tag
    print(f"\nCreating tag {tag}...")
    if not args.dry_run:
        run_command(["git", "tag", "-a", tag, "-m", f"Release {tag}"])
        run_command(["git", "push", "origin", tag])
        print(f"✓ Tag {tag} created and pushed")
    else:
        print(f"Would create tag: {tag}")

    print("\n✓ Release process complete!")
    print(f"\nGitHub Actions will now:")
    print("  1. Run tests on all platforms")
    print("  2. Build the package")
    print("  3. Publish to PyPI (if trusted publishing is configured)")
    print("  4. Create a GitHub release")

    if not args.dry_run:
        print(f"\nMonitor progress at: https://github.com/logandonley/mcp-mux/actions")


if __name__ == "__main__":
    main()
