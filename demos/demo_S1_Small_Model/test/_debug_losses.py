"""快速调试 losses.py 哪步在挂"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch

print("Step 1: import losses...", flush=True)
t = time.time()
from losses import cosine_feature_loss, freq_reg_loss, MultiEncoderProxyLoss
print(f"  done {time.time()-t:.1f}s", flush=True)

print("Step 2: cosine_feature_loss...", flush=True)
t = time.time()
f1, f2 = torch.rand(4, 512), torch.rand(4, 512)
l = cosine_feature_loss(f1, f2)
print(f"  done {time.time()-t:.1f}s, l={l.item():.4f}", flush=True)

print("Step 3: freq_reg_loss...", flush=True)
t = time.time()
orig = torch.rand(2, 3, 64, 64)
adv  = (orig + 0.02).clamp(0, 1)
l = freq_reg_loss(adv, orig)
print(f"  done {time.time()-t:.1f}s, l={l.item():.4f}", flush=True)

print("Step 4: MultiEncoderProxyLoss (no perceptual)...", flush=True)
t = time.time()
loss_fn = MultiEncoderProxyLoss({"encoder":1.0,"percept":0.1,"distort":0.5,"freq_reg":0.2}, use_perceptual=False)
total, ld = loss_fn(adv, orig, encoders=[], oracle_features={})
print(f"  done {time.time()-t:.1f}s, total={total.item():.4f}", flush=True)

print("All done!", flush=True)
