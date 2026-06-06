"""Produce a SAM3 instance-segmentation fine-tune config from the official roboflow full-ft template.

Edits (string-level, on the .250 official repo's config):
  - enable_segmentation: False -> True            (turn on the mask head + mask data)
  - add the `Masks` instance-mask loss (dice+focal) into the active loss_fns_find
  - paths -> our COCO data / bpe / log dir
  - supercategory -> literal "sam3_coco"; num_images -> null (use all crops)
  - model: load LOCAL sam3.pt (load_from_HF: false) instead of downloading
  - val /test/ -> /valid/ (our converter wrote train/ + valid/)
  - launcher gpus_per_node 2 -> 4; submitit use_cluster -> False (local 4-GPU)
"""
import sys

SRC = "/home/ps/sam3/sam3-official/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
DST = "/home/ps/sam3/sam3-official/sam3/train/configs/roboflow_v100/cropland_ft.yaml"

s = open(SRC).read()

# 1) enable segmentation (the single literal; the model section references ${scratch...})
assert s.count("enable_segmentation: False") == 1
s = s.replace("enable_segmentation: False", "enable_segmentation: True")

# 2) add instance-mask loss after the ACTIVE pad_scale_pos (8-space indent, no '#')
MASKS = """        pad_scale_pos: 1.0
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict:
          loss_mask: 200.0
          loss_dice: 10.0
        compute_aux: false"""
assert s.count("\n        pad_scale_pos: 1.0\n") == 1, "active pad_scale_pos not unique"
s = s.replace("\n        pad_scale_pos: 1.0\n", "\n" + MASKS + "\n", 1)

# 3) paths
s = s.replace("<YOUR_DATASET_DIR>", "/mnt/sda/zf/landform/data")
s = s.replace("<YOUR EXPERIMENET LOG_DIR>", "/mnt/sda/zf/landform/results/sam3_ft")
s = s.replace("<BPE_PATH>", "/home/ps/sam3/sam3-official/sam3/assets/bpe_simple_vocab_16e6.txt.gz")

# 4) dataset selection: literal supercategory + use all crops
s = s.replace("supercategory: ${all_roboflow_supercategories.${string:${submitit.job_array.task_index}}}",
              "supercategory: sam3_coco")
s = s.replace("num_images: 100", "num_images: null")

# 5) load LOCAL checkpoint (avoid HF download on the slow box)
s = s.replace(
    "enable_segmentation: ${scratch.enable_segmentation} # Warning: Enable this if using segmentation.",
    "enable_segmentation: ${scratch.enable_segmentation}\n"
    "    checkpoint_path: /home/ps/sam3/sam3_weights/sam3.pt\n"
    "    load_from_HF: false")

# 6) our converter wrote train/ + valid/ (template val uses /test/)
s = s.replace("/test/", "/valid/")

# 7) local 4-GPU
s = s.replace("gpus_per_node: 2", "gpus_per_node: 4")
s = s.replace("use_cluster: True", "use_cluster: False")

open(DST, "w").write(s)
print(f"wrote {DST}")
# echo the key changed lines for verification
for key in ["enable_segmentation:", "loss_fns.Masks", "checkpoint_path:", "load_from_HF:",
            "supercategory: sam3_coco", "num_images: null", "gpus_per_node:", "use_cluster:",
            "img_folder:", "ann_file:", "bpe_path:"]:
    for ln in s.splitlines():
        if key in ln:
            print("  ", ln.strip()); break
