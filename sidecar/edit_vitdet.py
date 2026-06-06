"""Patch the official SAM3 vitdet MLP so it trains: the fused `addmm_act` op detaches weights and
asserts grad is disabled (inference-only), which crashes fine-tuning. Use the eager path when grad
is enabled (trainable), keep the fast fused path for inference. Idempotent."""
F = "/home/ps/sam3/sam3-official/sam3/model/vitdet.py"
s = open(F).read()

old = "    def forward(self, x):\n        x = addmm_act(type(self.act), self.fc1, x)\n"
new = (
    "    def forward(self, x):\n"
    "        if torch.is_grad_enabled():\n"
    "            x = self.act(self.fc1(x))\n"
    "        else:\n"
    "            x = addmm_act(type(self.act), self.fc1, x)\n"
)
if new.strip() in s:
    print("already patched")
else:
    assert s.count(old) == 1, f"expected 1 match, got {s.count(old)}"
    s = s.replace(old, new)
    if "\nimport torch\n" not in ("\n" + s):
        s = "import torch\n" + s
    open(F, "w").write(s)
    print("patched vitdet MLP.forward: eager path when grad enabled")
