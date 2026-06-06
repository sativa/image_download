"""decoder._get_rpb_matrix: coord_cache returns coords on the device of the FIRST call (cpu during
build/warmup) but boxes_xyxy is cuda in training -> mismatch. Force cached coords to the boxes device."""
F = "/home/ps/sam3/sam3-official/sam3/model/decoder.py"
s = open(F).read()
old = "            coords_h, coords_w = self.coord_cache[feat_size]\n"
new = ("            coords_h, coords_w = self.coord_cache[feat_size]\n"
       "            coords_h = coords_h.to(boxes_xyxy.device)\n"
       "            coords_w = coords_w.to(boxes_xyxy.device)\n")
if "coords_h = coords_h.to(boxes_xyxy.device)" in s:
    print("already patched")
else:
    assert s.count(old) == 1, f"count {s.count(old)}"
    s = s.replace(old, new)
    open(F, "w").write(s)
    print("patched decoder _get_rpb_matrix: coords -> boxes_xyxy.device")
