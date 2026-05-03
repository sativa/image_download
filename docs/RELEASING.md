# Releasing

Imagery Downloader uses tag-triggered GitHub Actions for releases.

## Cut a release

1. Make sure `main` is green on CI.
2. Bump `src-tauri/Cargo.toml` version (or rely on tag injection — both work; the release workflow rewrites the version from the tag before building).
3. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. Watch the **Release** workflow in GitHub Actions. Expect ~20 min for all
   three legs (macOS arm64, macOS x64, Windows x64).
5. Open the auto-created **draft** release on GitHub. Review that the
   artifact list matches the matrix:
   - `Imagery.Downloader_<ver>_aarch64.dmg`
   - `Imagery.Downloader_<ver>_aarch64.app.tar.gz`
   - `Imagery.Downloader_<ver>_x64.dmg`
   - `Imagery.Downloader_<ver>_x64.app.tar.gz`
   - `Imagery.Downloader_<ver>_x64_en-US.msi`
   - `Imagery.Downloader_<ver>_x64-setup.exe`

   That's six files total.
6. Edit the release notes inline, then click **Publish**.

## Pre-release / RC

Tags containing `-` are automatically marked as pre-release:

```bash
git tag v0.2.0-rc1
git push origin v0.2.0-rc1
```

GitHub UI shows them with a "Pre-release" badge.

## Roll back

To unpublish a release and remove its tag:

UI flow:
1. **Releases** → … menu on the release → **Delete release**.
2. Confirm "Also delete the tag" if shown.

Or command line:

```bash
gh release delete v0.2.0 --yes
git push origin :refs/tags/v0.2.0
git tag -d v0.2.0
```

The next tag pushed afterwards will produce a fresh release.

## Versioning

We follow SemVer:
- `0.x.y`: scaffold + early development. Breaking changes can land in minor.
- `1.x.y`: stable IPC contract. Breaking changes only in major.

The IPC contract is defined by `src/lib/types.ts` plus the event channel
names in `src/lib/ipc.ts`. Any change there is an IPC-contract change.
