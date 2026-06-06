"""Unified poll-based GPU dispatcher for ALL remaining jobs after the clean reset:
  - route_a (1m-native 15ch)             [priority]
  - c_stage2_esri / c_stage2_google      (dual-vs-single stage-2; feat_esri/google already inferred)
  - 4 learning-curve jobs (1k/2.5k x baseline/route-c)
Launches each on any GPU that is currently free (util<20 & mem<2.5GB), <=1 job/GPU.
No hard-coded GPU pinning -> no collisions. route_a is first in the queue."""
import os, subprocess, time
from pathlib import Path

LF = "/mnt/sda/zf/landform"; SD = "/home/ps/landform/sidecar"; D = "/home/ps/landform/data"
PY = os.path.expanduser("~/miniconda3/bin/python")
FEAT = f"{LF}/data/c_stage1_feat"

JOBS = [
    ("route_a", ["train_route_a.py", "--data-dir", f"{LF}/data/c_1m", "--out", f"{LF}/results/route_a", "--epochs", "20", "--workers", "10"]),
    ("lc_bl_1k", ["train_c_stage2.py", "--regions-json", f"{D}/v40_1000.json", "--no-1m", "--seed", "0", "--out-dir", f"{LF}/results/lc_bl_1k"]),
    ("lc_rc_1k", ["train_c_stage2.py", "--regions-json", f"{D}/v40_1000.json", "--feat-dir", FEAT, "--seed", "0", "--out-dir", f"{LF}/results/lc_rc_1k"]),
    ("lc_bl_2k5", ["train_c_stage2.py", "--regions-json", f"{D}/v40_2500.json", "--no-1m", "--seed", "0", "--out-dir", f"{LF}/results/lc_bl_2k5"]),
    ("lc_rc_2k5", ["train_c_stage2.py", "--regions-json", f"{D}/v40_2500.json", "--feat-dir", FEAT, "--seed", "0", "--out-dir", f"{LF}/results/lc_rc_2k5"]),
]


def gpu_free(g):
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits", "-i", str(g)],
                           capture_output=True, text=True, timeout=15).stdout.strip()
        u, m = [int(x.strip()) for x in o.split(",")]
        return u < 20 and m < 2500
    except Exception:
        return False


def launch(name, args, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, "-u", f"{SD}/{args[0]}"] + args[1:] + ["--device", "cuda:0"]
    lf = open(f"{LF}/results/{name}.log", "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=SD)


def main():
    running = {}; pending = list(JOBS)
    print(f"[extra-dispatch] {len(pending)} jobs, poll-based (route_a priority)", flush=True)
    while pending or running:
        for g, (nm, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[done] {nm} gpu{g} rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
                del running[g]
        for g in (0, 1, 2, 3):
            if not pending or g in running:
                continue
            if not gpu_free(g):
                continue
            nm, args = pending.pop(0)
            running[g] = (nm, launch(nm, args, g))
            print(f"[launch] {nm} -> gpu{g} {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(15)  # let it claim the GPU before the next free-check
        time.sleep(25)
    print(f"[extra-dispatch] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
