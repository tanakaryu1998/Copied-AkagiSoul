Akagi 3.8.0 — portable build (macos-arm64)

Quick start
-----------
1. Move this folder anywhere you have write permission (e.g. ~/Apps/, Desktop, etc.).
2. Run the binary:
     Linux/macOS:  ./akagi
     Windows:      akagi.exe
3. On first launch, Akagi creates these directories alongside the binary:
     config.toml   logs/   history/   ca/   mjai_bot/

Platform notes
--------------
- The binary is unsigned. macOS Gatekeeper will block the first launch.
  Either run once with the quarantine bit removed:
    xattr -cr "$(pwd)/akagi-3.8.0-macos-arm64"
  or right-click the binary and choose "Open" the first time.
- Apple Silicon only — no Intel build.

Full documentation: https://github.com/shinkuan/AkagiV3
