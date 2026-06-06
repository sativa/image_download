import rasterio, glob
from rasterio.merge import merge
files = sorted(glob.glob("/mnt/sda/zf/landform/data/yz_cont_tif/*_esri.tif"))
srcs = [rasterio.open(f) for f in files]
mosaic, tr = merge(srcs)
meta = srcs[0].meta.copy()
meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tr, compress="deflate")
out = "/mnt/sda/zf/landform/data/yz_cont_merged.tif"
with rasterio.open(out, "w", **meta) as d:
    d.write(mosaic)
print(f"merged {len(files)} tif -> {mosaic.shape[2]}x{mosaic.shape[1]} px -> {out}")
