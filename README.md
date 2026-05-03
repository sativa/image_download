# Imagery Downloader

Standalone extraction of **Step 11 (影像下载)** from the Gansu soil-mapping
pipeline (`/Volumes/Thunderbolt3/gansu10m/run_pipeline.py`). Downloads
high-resolution satellite imagery (ESRI / Microsoft Bing / Google) for the
polygons in a GeoPackage and writes one GeoTIFF per polygon, clipped to the
polygon's extent.

## Files

| File | Purpose |
| --- | --- |
| `download_imagery.py` | Main downloader (was `download_google_imagery.py`). |
| `monitor_download.py` | Poll a running download and print progress. |
| `verify_polygon_imagery.py` | Verify each polygon was paired with a non-empty image. |
| `visualize_imagery.py` | Render a PNG preview + metadata panel for one GeoTIFF. |
| `requirements.txt` | Python dependencies. |

## Install

```bash
pip install -r requirements.txt
```

`rasterio` and `geopandas` need GDAL — install via conda if pip fails:

```bash
conda install -c conda-forge gdal geopandas rasterio
```

## Quick start

```bash
python download_imagery.py path/to/polygons.gpkg \
    --output-dir ./out/imagery \
    --zoom 17 \
    --source google \
    --max-workers 50
```

The script reprojects to WGS84, picks the fastest available source if
`--source auto`, computes the tile range covering each polygon, fetches all
tiles in parallel, mosaics + clips to the polygon, and writes
`polygon_<idx>_zoom<z>.tif` plus `download_summary.json`.

## Common options (mirrors Step 11 in `run_pipeline.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--zoom` | 22 | Tile zoom level (17 ≈ 0.96 m/px, 22 ≈ 0.03 m/px). |
| `--source` | auto | `esri` / `bing` / `google` / `auto`. |
| `--no-auto-zoom` | off | Disable auto zoom adjustment to satisfy `--min-resolution`. |
| `--min-resolution` | 5.0 | Target m/px; auto-zoom raises zoom until met. |
| `--min-area` | None | Skip polygons under this area (hectares). |
| `--max-area` | None | Skip polygons over this area (km²). |
| `--change-type` | None | Only process polygons whose `Change_Type` field equals this. |
| `--change-types` | None | Same, but a list. |
| `--max-workers` | 50 | Concurrent download threads. |
| `--max-polygons` | None | Cap (testing). |

`Ctrl-C` is handled cleanly: in-flight requests are aborted and a summary is
still written.

## Output

```
<output-dir>/
  polygon_0_zoom17.tif
  polygon_1_zoom17.tif
  ...
  download_summary.json                   # status per polygon
  <input-stem>_with_imagery.gpkg          # original GPKG + image_path column
```

## Helper utilities

```bash
# Watch progress while a download runs (edit paths in the script first)
python monitor_download.py

# Spot-check that polygons line up with their imagery
python verify_polygon_imagery.py <gpkg-with-image_path>

# Render a single preview PNG
python visualize_imagery.py            # edits the path inside the script
```

## Mapping back to Step 11

The Step 11 wrapper in `run_pipeline.py` (`run_download_imagery`) just builds
this CLI invocation and runs it as a subprocess. The parameter dict
`IMAGERY_DOWNLOAD_PARAMS` maps 1:1 to the flags above:

| `IMAGERY_DOWNLOAD_PARAMS` key | CLI flag |
| --- | --- |
| `gpkg_file` | positional `input_gpkg` |
| `zoom` | `--zoom` |
| `source` | `--source` |
| `auto_zoom` | absence of `--no-auto-zoom` |
| `change_type` | `--change-type` |
| `min_area` / `max_area` | `--min-area` / `--max-area` |
| `max_workers` | `--max-workers` |
| `output_dir` | `--output-dir` |
