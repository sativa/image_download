"""Fill all idle GPUs with stage-2 ensemble members (baseline 10m + route-c 10m+1m),
coexisting with the running run_route_c_full.sh orchestrator.

GPU pool grows as the orchestrator frees GPUs:
  always {1,3}; +{0} once infer done (feat==5120); +{2} once orchestrator's
  route-c seed0 finishes (c_stage2_1m/final.json exists).
Members (all efficientnet-b5, weights cached): unet seed1/seed2 + deeplabv3plus seed0,
for both baseline (--no-1m) and route-c (real 1m feat). seed0-unet of each is produced
by the orchestrator (c_stage2_base / c_stage2_1m), so the full ensembles are
{seed0,seed1,seed2,deeplab} x {baseline, route-c}.
"""
import glob, os, subprocess, time
from pathlib import Path

LF = "/mnt/sda/zf/landform"
SD = "/home/ps/landform/sidecar"
PY = os.path.expanduser("~/miniconda3/bin/python")
REG = "/home/ps/landform/data/v40_5k.json"
FEAT = f"{LF}/data/c_stage1_feat"


def n_feat():
    return len(glob.glob(f"{FEAT}/*.npz"))


def s0_done():
    return Path(f"{LF}/results/c_stage2_1m/final.json").exists()


def gpu_free(g):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(g)],
            capture_output=True, text=True, timeout=15).stdout.strip()
        u, m = [int(x.strip()) for x in out.split(",")]
        return u < 20 and m < 2500
    except Exception:
        return False


# (name, needs_feat, extra_args)
JOBS = [
    ("bl_s1", False, ["--no-1m", "--seed", "1"]),
    ("bl_s2", False, ["--no-1m", "--seed", "2"]),
    ("bl_dl", False, ["--no-1m", "--seed", "0", "--arch", "deeplabv3plus"]),
    ("rc_s1", True, ["--feat-dir", FEAT, "--seed", "1"]),
    ("rc_s2", True, ["--feat-dir", FEAT, "--seed", "2"]),
    ("rc_dl", True, ["--feat-dir", FEAT, "--seed", "0", "--arch", "deeplabv3plus"]),
]


def launch(name, args, gpu):
    out = f"{LF}/results/ens_{name}"
    cmd = [PY, "-u", f"{SD}/train_c_stage2.py", "--regions-json", REG,
           "--out-dir", out, "--device", "cuda:0", "--workers", "12"] + args
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    lf = open(f"{LF}/results/ens_{name}.log", "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=SD)


def main():
    running = {}          # gpu -> (name, popen)
    pending = list(JOBS)
    print(f"[dispatch] {len(pending)} ensemble members; fills GPUs as freed", flush=True)
    while pending or running:
        for g, (nm, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[done] {nm} gpu{g} rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
                del running[g]
        pool = [1, 3]
        if n_feat() >= 5120:
            pool.append(0)
        if s0_done():
            pool.append(2)
        feat_ready = n_feat() >= 5120
        for g in pool:
            if not pending:
                break
            if g in running:
                continue
            idx = next((i for i, (nm, nf, a) in enumerate(pending) if (not nf) or feat_ready), None)
            if idx is None:
                break
            if not gpu_free(g):
                continue
            nm, nf, a = pending.pop(idx)
            running[g] = (nm, launch(nm, a, g))
            print(f"[launch] {nm} -> gpu{g} (feat_ready={feat_ready}) {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(10)
        time.sleep(25)
    print(f"[dispatch] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
