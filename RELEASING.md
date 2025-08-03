# Release Process

Version numbers come from git tags (via setuptools-scm)

## Creating a Release

Use the release script:

```bash
# Bump patch version (0.1.0 → 0.1.1)
python scripts/release.py --patch

# Bump minor version (0.1.1 → 0.2.0)
python scripts/release.py --minor

# Bump major version (0.2.0 → 1.0.0)
python scripts/release.py --major

# Specific version
python scripts/release.py 0.2.0

# Dry run (see what would happen)
python scripts/release.py --patch --dry-run
```

The script will:

1. Check that working directory is clean
2. Pull latest changes
3. Run tests
4. Create and push the tag
5. Trigger GitHub Actions

## What Happens Next

Once you push a tag, GitHub Actions automatically:

1. **Tests** the code on Ubuntu
2. **Builds** the package (version from git tag)
3. **Publishes** to PyPI using trusted publishing
4. **Creates** a GitHub release with artifacts

Monitor progress at: https://github.com/logandonley/mcp-mux/actions

## Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

Between releases, setuptools-scm generates development versions like `0.2.1.dev5+g1234567`.
