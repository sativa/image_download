#!/bin/bash
# Reliable reset of the GPU0/1 side ONLY. Preserves the ablation stage-2 (v40_5k on GPU2/3)
# and its orchestrator. Kills the dispatcher + any route_a + the LC jobs (v40_1000/2500),
# then relaunches the unified 5-job poll dispatcher fully detached.
L=/mnt/sda/zf/landform
pkill -f run_extra_dispatch; pkill -f train_route_a; pkill -f v40_1000; pkill -f v40_2500
sleep 5
cd /home/ps/landform/sidecar
setsid ~/miniconda3/bin/python -u run_extra_dispatch.py > "$L/results/extra_dispatch.log" 2>&1 < /dev/null &
sleep 4
echo "RESET ok | dispatcher=$(pgrep -fc run_extra_dispatch) route_a=$(pgrep -fc train_route_a) ablation_v40_5k=$(pgrep -fc v40_5k) orch=$(pgrep -fc run_source_ablation)"
echo "dispatch head: $(head -1 "$L/results/extra_dispatch.log")"
