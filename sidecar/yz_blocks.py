"""Seamless full-county vectorization via N×N-cell BLOCK mosaics, in-process model, WHOLE-BLOCK
FFL polygonization (no crop -> FFL does not collapse). Within a block: one coherent inference +
one watershed -> zero internal cell seams, full coverage. Only block-to-block borders seam (1/N).
Reuses the proven _infer_all. Blocks fan across 4 GPUs (spawn pool), model resident per worker."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
SIDECAR="/home/ps/landform/sidecar"; _M={}

def _init(gpus, weights, backbone, counter, bboxes, halo_m):
    with counter.get_lock():
        wid=counter.value; counter.value+=1
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus[wid%len(gpus)])
    import sys
    if SIDECAR not in sys.path: sys.path.insert(0,SIDECAR)
    import torch
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDDF
    d3=AutoModel.from_pretrained(backbone, local_files_only=True)
    m=DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).cuda()
    sd=torch.load(weights, map_location="cuda", weights_only=True); msd=m.state_dict()
    m.load_state_dict({k:v for k,v in sd.items() if k in msd and msd[k].shape==v.shape}, strict=False)
    m.eval(); _M["model"]=m; _M["bboxes"]=bboxes; _M["halo_m"]=halo_m
    print("[worker %d] resident on GPU %s"%(wid,os.environ["CUDA_VISIBLE_DEVICES"]), flush=True)

def _infer_all(model, x6, cs=448):
    import torch, torch.nn.functional as F
    from train_dino_1m import norm6
    _,H,W=x6.shape; ndvi=np.zeros((5,H,W),np.float32)
    acc=np.zeros((9,H,W),np.float32); accd=np.zeros((H,W),np.float32); accb=np.zeros((H,W),np.float32)
    accf=np.zeros((4,H,W),np.float32); cnt=np.zeros((H,W),np.float32); st=max(1,cs//2)
    ys=list(range(0,max(1,H-cs+1),st)); xs=list(range(0,max(1,W-cs+1),st))
    if ys[-1]!=H-cs: ys.append(max(0,H-cs))
    if xs[-1]!=W-cs: xs.append(max(0,W-cs))
    win=np.maximum(np.outer(np.hanning(cs),np.hanning(cs)),1e-3).astype(np.float32)
    for t in ys:
        for l in xs:
            xc=np.concatenate([norm6(x6[:,t:t+cs,l:l+cs]), ndvi[:,t:t+cs,l:l+cs]],0)
            xb=torch.from_numpy(xc).unsqueeze(0).cuda()
            with torch.amp.autocast("cuda",dtype=torch.bfloat16), torch.no_grad():
                o=model(xb); cl,bn,ds,ff=o[0],o[1],o[2],o[3]
                if cl.shape[-2:]!=(cs,cs):
                    cl=F.interpolate(cl,(cs,cs),mode="bilinear",align_corners=False); bn=F.interpolate(bn,(cs,cs),mode="bilinear",align_corners=False)
                    ds=F.interpolate(ds,(cs,cs),mode="bilinear",align_corners=False); ff=F.interpolate(ff,(cs,cs),mode="bilinear",align_corners=False)
                pr=torch.softmax(cl.float(),1)[0].cpu().numpy(); pd=torch.sigmoid(ds.float())[0,0].cpu().numpy()
                pb=torch.sigmoid(bn.float())[0,0].cpu().numpy(); pf=ff.float()[0].cpu().numpy()
            acc[:,t:t+cs,l:l+cs]+=pr*win; accd[t:t+cs,l:l+cs]+=pd*win; accb[t:t+cs,l:l+cs]+=pb*win
            accf[:,t:t+cs,l:l+cs]+=pf*win; cnt[t:t+cs,l:l+cs]+=win
    cnt=np.maximum(cnt,1e-6); accf/=cnt
    return acc/cnt, accd/cnt, accb/cnt, accf[0]+1j*accf[1], accf[2]+1j*accf[3]

class _P: min_dist=20; peak_thr=0.4; min_area_px=200; ridge=False; downscale=1; smooth_iters=1

def _process(task):
    block_id, cell_list, tif_dir, out_dir = task
    import sys
    if SIDECAR not in sys.path: sys.path.insert(0,SIDECAR)
    import rasterio
    from rasterio.merge import merge
    import geopandas as gpd
    from dino_parcel_export import build_idmap, NAME_ZH, NAME_EN, HEX
    from ff_polygonize import polygonize_ff
    op=Path(out_dir)/("%s.parquet"%block_id)
    if op.exists(): return (block_id,-1,"skip")
    t0=time.time()
    try:
        bboxes=_M["bboxes"]; halo=_M["halo_m"]
        cb=[bboxes[c] for c in cell_list if c in bboxes]
        if not cb: return (block_id,-3,"no_tif")
        core=(min(b[0] for b in cb),min(b[1] for b in cb),max(b[2] for b in cb),max(b[3] for b in cb))
        exp=(core[0]-halo,core[1]-halo,core[2]+halo,core[3]+halo)  # >=30% overlap halo: each block sees neighbour context at its borders
        nb=[c for c,b in bboxes.items() if not (b[2]<=exp[0] or b[0]>=exp[2] or b[3]<=exp[1] or b[1]>=exp[3])]
        files=[str(Path(tif_dir)/("%s_esri.tif"%c)) for c in nb if (Path(tif_dir)/("%s_esri.tif"%c)).exists()]
        if not files: return (block_id,-3,"no_tif")
        srcs=[rasterio.open(f) for f in files]; crs=srcs[0].crs
        mosaic,tr=merge(srcs, bounds=exp); [s.close() for s in srcs]
        rgb=np.ascontiguousarray(mosaic[:3]).astype(np.uint8); x6=np.concatenate([rgb,rgb],0)
        cls,dist,bnd,c0,c2=_infer_all(_M["model"], x6)            # core+halo: Hanning-blended overlapping windows -> consistent borders
        idmap,cls_of=build_idmap(cls,dist,bnd,_P())
        from rasterio.windows import from_bounds as _fb, transform as _wtr, Window as _Win
        win=_fb(core[0],core[1],core[2],core[3], tr)              # crop idmap back to CORE -> exact tiling, zero output overlap
        r0=max(0,int(round(win.row_off))); c0p=max(0,int(round(win.col_off)))
        hh=int(round(win.height)); ww=int(round(win.width))
        idmap=np.ascontiguousarray(idmap[r0:r0+hh, c0p:c0p+ww]); tr=_wtr(_Win(c0p,r0,ww,hh), tr)
        from rasterio.features import shapes as _shapes
        from shapely.geometry import shape as _shape
        idmap=idmap.astype(np.int32); rows=[]
        for geom,val in _shapes(idmap, mask=idmap>0, transform=tr):   # exact partition: full cover, no overlap/overflow
            c=cls_of.get(int(val))
            if not c: continue
            g=_shape(geom).simplify(1.5, preserve_topology=True)
            if not g.is_valid: g=g.buffer(0)
            if g.is_empty: continue
            if g.geom_type=="Polygon": rows.append({"parcel_id":int(val),"class_id":c,"geometry":g})
            else:
                for pp in getattr(g,"geoms",[]):
                    if getattr(pp,"geom_type","")=="Polygon" and not pp.is_empty and pp.area>0: rows.append({"parcel_id":int(val),"class_id":c,"geometry":pp})
        if not rows:
            gpd.GeoDataFrame({"geometry":[]},crs="EPSG:4326").to_parquet(op); return (block_id,0,"%.0fs"%(time.time()-t0))
        gdf=gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        gdf["area_m2"]=gdf.to_crs("EPSG:32648").geometry.area.round(1).values
        gdf["label"]=[NAME_ZH[c] for c in gdf["class_id"]]; gdf["label_en"]=[NAME_EN[c] for c in gdf["class_id"]]
        gdf["rgb_hex"]=[HEX[c] for c in gdf["class_id"]]; gdf["block"]=block_id
        gdf=gdf.to_crs("EPSG:4326"); gdf.to_parquet(op)
        return (block_id, len(gdf), "%.0fs n=%d"%(time.time()-t0,len(files)))
    except Exception as ex:
        import traceback; traceback.print_exc(); return (block_id,-2,str(ex)[:200])

def _bbox(args):
    cell, tif_dir = args
    import rasterio
    f=Path(tif_dir)/("%s_esri.tif"%cell)
    if not f.exists(): return (cell,None)
    with rasterio.open(f) as d: return (cell, tuple(d.bounds))

def _cluster(vals, tol):
    s=sorted(set(vals)); groups=[[s[0]]]
    for v in s[1:]:
        if v-groups[-1][-1]<tol: groups[-1].append(v)
        else: groups.append([v])
    m={}
    for gi,g in enumerate(groups):
        for v in g: m[v]=gi
    return m

def build_blocks(bboxes, N):
    lefts=[b[0] for b in bboxes.values()]; bots=[b[1] for b in bboxes.values()]
    cw=np.median([b[2]-b[0] for b in bboxes.values()]); chh=np.median([b[3]-b[1] for b in bboxes.values()])
    cmap=_cluster(lefts, 0.5*cw); rmap=_cluster(bots, 0.5*chh)
    blocks={}
    for c,b in bboxes.items():
        col=cmap[b[0]]; row=rmap[b[1]]
        blocks.setdefault((col//N, row//N), []).append(c)
    return blocks

def main():
    import multiprocessing as mp
    ap=argparse.ArgumentParser()
    ap.add_argument("--regions", default="/tmp/yz_full_regions.json")
    ap.add_argument("--tif-dir", default="/mnt/sda/zf/landform/data/yz_full_tif")
    ap.add_argument("--out-dir", default="/mnt/sda/zf/landform/results/yz_blocks")
    ap.add_argument("--county-out", default="/mnt/sda/zf/landform/results/yuzhong_blocks_region.parquet")
    ap.add_argument("--weights", default="/mnt/sda/zf/landform/results/dino_v3_bddf/last.pt")
    ap.add_argument("--backbone", default="/home/ps/landform/dinov3/dinov3-vitl16-sat493m")
    ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--block", type=int, default=2); ap.add_argument("--smoke-cell", default=""); ap.add_argument("--halo-cells", type=float, default=1.0)
    a=ap.parse_args()
    gpus=[int(g) for g in a.gpus.split(",")]
    out_dir=Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names=["%s_%s"%(c["county"],c["idx"]) for c in json.loads(Path(a.regions).read_text())]
    ctx=mp.get_context("spawn"); t0=time.time()
    with ctx.Pool(min(32,os.cpu_count())) as pp:
        bbl=pp.map(_bbox, [(n,a.tif_dir) for n in names])
    bboxes={c:bb for c,bb in bbl if bb is not None}
    blocks=build_blocks(bboxes, a.block)
    _cw=float(np.median([b[2]-b[0] for b in bboxes.values()])); _halo=a.halo_cells*_cw
    if a.smoke_cell:
        cb=bboxes[a.smoke_cell]
        sel=[(k,v) for k,v in blocks.items() if a.smoke_cell in v]
        blocks=dict(sel)
    tasks=[("b%02d_%02d"%(k[0],k[1]), v, a.tif_dir, str(out_dir)) for k,v in sorted(blocks.items())]
    nproc=len(gpus)*a.per_gpu
    print("[blk] %d cells -> %d blocks (block=%dx%d), %d workers"%(len(bboxes),len(tasks),a.block,a.block,nproc), flush=True)
    counter=ctx.Value("i",0); ok=done=bad=0
    pool=ctx.Pool(nproc, initializer=_init, initargs=(gpus,a.weights,a.backbone,counter,bboxes,_halo))
    try:
        for bid,n,msg in pool.imap_unordered(_process, tasks):
            done+=1
            if n>=0: ok+=1
            elif n in (-2,-3): bad+=1; print("  %s FAIL %s"%(bid,msg), flush=True)
            r=(time.time()-t0)/max(done,1)
            print("  [%d/%d] ok=%d bad=%d %s:%s | %.0fs/blk | ETA %.0fmin"%(done,len(tasks),ok,bad,bid,msg,r,r*(len(tasks)-done)/60), flush=True)
    finally:
        pool.terminate()
    import geopandas as gpd, pandas as pd
    gdfs=[]
    for bid,_,_,_ in tasks:
        p=out_dir/("%s.parquet"%bid)
        if not p.exists(): continue
        try:
            d=gpd.read_parquet(p)
            if len(d): gdfs.append(d)
        except Exception as ex: print("  concat skip %s: %s"%(bid,ex), flush=True)
    if gdfs:
        reg=gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        reg.insert(0,"gid",range(1,len(reg)+1)); reg.to_parquet(a.county_out)
        from collections import Counter
        cc=Counter(reg["label"]); ar=reg.groupby("label")["area_m2"].sum().div(1e6).round(1)
        print("[blk] DONE ok=%d bad=%d | %d parcels -> %s"%(ok,bad,len(reg),a.county_out), flush=True)
        print("  counts: %s"%dict(cc), flush=True); print("  km2: %s"%ar.to_dict(), flush=True)
    print("  total %.0fs"%(time.time()-t0), flush=True)
    os._exit(0)

if __name__=="__main__": main()
