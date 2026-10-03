"""Time real training steps (real data, real loader) per model and batch size; prints minutes per epoch."""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceshield.data import ASVspoofLA  # noqa: E402
from voiceshield.models import build  # noqa: E402

root = sys.argv[1]
tr = ASVspoofLA("train", train=True, root=Path(root))
for name, bs in [("lcnn", 64), ("rawnet2", 32), ("aasist_l", 24), ("aasist", 24)]:
    torch.cuda.empty_cache()
    m = build(name).cuda().train()
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    sc = torch.cuda.amp.GradScaler()
    it = iter(DataLoader(tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True))
    torch.cuda.reset_peak_memory_stats()
    n = 30
    for i in range(n + 5):
        if i == 5:
            torch.cuda.synchronize(); t = time.time()
        x, y, _ = next(it)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(m(x.cuda()).float(), y.cuda())
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    torch.cuda.synchronize()
    st = (time.time() - t) / n
    print(json.dumps({"model": name, "bs": bs, "s_per_step": round(st, 3), "min_per_epoch": round(len(tr) / bs * st / 60, 1),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
    del m, opt
