"""Scoped device fix: global set_default_device broke the DataLoader sampler's cpu generator.
Instead, revert the global setting and wrap ONLY the model forward + loss in `with torch.device(cuda)`
so the model's default-cpu tensor creations land on cuda, while the DataLoader (outside) keeps cpu."""
F = "/home/ps/sam3/sam3-official/sam3/train/trainer.py"
s = open(F).read()

# 1) revert the global set_default_device (added earlier in _setup_device)
s = s.replace('\n            torch.set_default_device(self.device)', '')

# 2) wrap the forward + back_convert + loss in a device context
old = ('        find_stages = model(batch)\n'
       '        find_targets = [\n'
       '            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets\n'
       '        ]\n'
       '        batch_size = len(batch.img_batch)\n'
       '        loss = self._find_loss(key)(find_stages, find_targets)')
new = ('        with torch.device(self.device):\n'
       '            find_stages = model(batch)\n'
       '            find_targets = [\n'
       '                unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets\n'
       '            ]\n'
       '            batch_size = len(batch.img_batch)\n'
       '            loss = self._find_loss(key)(find_stages, find_targets)')

if "with torch.device(self.device):" in s:
    print("already wrapped")
else:
    assert s.count(old) == 1, f"count {s.count(old)}"
    s = s.replace(old, new)
    open(F, "w").write(s)
    print("reverted global set_default_device + wrapped forward/loss in device context")
