"""Fix SAM3 trainer device bug: in _get_img_feats, the backbone's vision_pos_enc is created on CPU
while backbone_fpn (real features) and img_ids are on CUDA -> index error. Force img_ids and
vision_pos_enc onto the features' device for consistency. Idempotent."""
F = "/home/ps/sam3/sam3-official/sam3/model/sam3_image.py"
s = open(F).read()

old1 = ('        if "backbone_fpn" in backbone_out:\n'
        '            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:')
new1 = ('        if "backbone_fpn" in backbone_out:\n'
        '            _dev = backbone_out["backbone_fpn"][0].device\n'
        '            img_ids = img_ids.to(_dev)\n'
        '            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:')

old2 = '            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]'
new2 = '            vis_pos_enc = [x.to(_dev) for x in backbone_out["vision_pos_enc"][-self.num_feature_levels :]]'

if "_dev = backbone_out" in s:
    print("already patched")
else:
    assert s.count(old1) == 1, f"old1 count {s.count(old1)}"
    assert s.count(old2) == 1, f"old2 count {s.count(old2)}"
    s = s.replace(old1, new1).replace(old2, new2)
    open(F, "w").write(s)
    print("patched _get_img_feats: force img_ids + vision_pos_enc to features device")
