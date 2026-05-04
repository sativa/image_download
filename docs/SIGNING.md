# Code Signing

This project ships **unsigned binaries by default**. Signing is wired in
`.github/workflows/release.yml` but inactive until the corresponding
secrets are populated.

## Current behaviour

### macOS

- Unsigned `.app` triggers Gatekeeper. First-launch UX:
  1. User downloads `.dmg`, drags `.app` to `/Applications`.
  2. Double-click → "App is from an unidentified developer." → user must
     **right-click → Open** the first time.
  3. Subsequent launches work normally.
- Some users report the binary is "damaged" after download. Fix:

  ```bash
  xattr -cr "/Applications/Imagery Downloader.app"
  ```

### Windows

- Unsigned `.msi` and `-setup.exe` trigger SmartScreen. First-launch UX:
  1. User downloads, double-clicks installer.
  2. SmartScreen says "Windows protected your PC". User clicks
     **More info → Run anyway**.
  3. Once installed, app launches without further warnings.

## Enabling Apple Developer ID signing (future)

Required:
- Apple Developer Program membership ($99/yr).
- A "Developer ID Application" certificate exported as `.p12`.
- An app-specific password for `notarytool` (created in Apple ID account
  settings).

Steps:
1. Add the following GitHub repository secrets:
   - `APPLE_CERTIFICATE` — base64 of the `.p12`
   - `APPLE_CERTIFICATE_PASSWORD` — password protecting the `.p12`
   - `APPLE_SIGNING_IDENTITY` — full identity string, e.g.
     `Developer ID Application: Your Name (TEAMID)`
   - `APPLE_ID` — Apple ID email
   - `APPLE_PASSWORD` — app-specific password
   - `APPLE_TEAM_ID` — 10-char team ID
2. Tauri-action picks up these env vars automatically — no conf change
   needed. It will sign **and** notarize.
3. Cut a new release.

## Enabling Windows code signing (future)

Required:
- A code-signing certificate (OV ~$200/yr or EV ~$300+/yr) as `.pfx`.

Steps:
1. Add secrets:
   - `WINDOWS_CERTIFICATE` — base64 of `.pfx`
   - `WINDOWS_CERTIFICATE_PASSWORD` — password
2. Wire them into `release.yml`'s `tauri-action` env block.
3. In `tauri.conf.json` under `bundle.windows`, set `certificateThumbprint`
   from the `.pfx`'s SHA-1, or rely on tauri-action env injection.
4. Cut a new release.

## Important: env vars must NOT be set when secrets are missing

`tauri-action` interprets a non-empty `APPLE_CERTIFICATE` env var as
"please import this certificate into the keychain". Even an empty string
from an unset GitHub secret will trigger the import attempt and fail with
`failed to import keychain certificate`. The current `release.yml` solves
this by **not setting any APPLE_* env at all**. When you enable signing,
re-add the env block — the secret values themselves will be non-empty.

## Why not now

- Apple Developer Program is $99/yr — defer until a public release is
  imminent.
- Windows OV cert is ~$200/yr; EV is ~$300+/yr. Same rationale.
- Until then, the documented Gatekeeper / SmartScreen workarounds are
  acceptable for a small user base.
- The release workflow's secret slots are already populated with `null`
  values via `${{ secrets.APPLE_* }}`, so adding real secrets later
  requires no workflow edit.
