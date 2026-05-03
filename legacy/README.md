# Legacy Python CLI

This directory preserves the original Python CLI from before the Tauri rewrite.
**It is not built or tested by CI.** Kept for reference only.

Original entry points:
- `download_imagery.py` — main CLI
- `monitor_download.py` — progress poller
- `verify_polygon_imagery.py` — output QA
- `visualize_imagery.py` — visualization helper

To run:

    cd legacy
    pip install -r requirements.txt
    python download_imagery.py --help

The new Tauri app supersedes all of the above.
