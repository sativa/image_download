"""Robust fix for SAM3 trainer cpu/cuda scatter: the model's forward creates several tensors on the
default (cpu) device (vision_pos_enc, decoder RPB matrix, ...) while params/data are on cuda. Set the
default device to cuda in the (main) training process so all default-device creations land on cuda.
DataLoader workers are separate processes (default cpu) so pin_memory still works. Idempotent."""
F = "/home/ps/sam3/sam3-official/sam3/train/trainer.py"
s = open(F).read()
old = ('            self.device = torch.device("cuda", self.local_rank)\n'
       '            torch.cuda.set_device(self.local_rank)')
new = old + '\n            torch.set_default_device(self.device)'
if "torch.set_default_device(self.device)" in s:
    print("trainer already patched")
else:
    assert s.count(old) == 1, f"count {s.count(old)}"
    s = s.replace(old, new)
    open(F, "w").write(s)
    print("patched trainer _setup_device: torch.set_default_device(cuda)")
