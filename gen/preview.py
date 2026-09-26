"""Rasterize loop frames from gen/data to PNG (linear interpolation, no easing)."""
import numpy as np
from PIL import Image, ImageDraw
import banner as B

dots = np.load(B.DATA / "dots.npy"); labels = np.load(B.DATA / "bands.npy")
P, L1, L2, L3 = np.load(B.DATA / "travellers.npy")
ys, xs = np.nonzero(dots)
kt = np.array([float(k) for k in B.loop_key_times().split(";")]) * B.LOOP

def seg(t):
    i = min(np.searchsorted(kt, t, side="right") - 1, len(kt) - 2)
    return i, (t - kt[i]) / (kt[i + 1] - kt[i])

def frame(t, S=2):
    i, f = seg(t)
    img = Image.new("RGB", (B.GW * S, B.GH * S), (10, 16, 31)); d = ImageDraw.Draw(img)
    bop = [1, 1, 0, 0, 0, 0, 0, 0, 1]; top = [0, 0, 1, 1, 1, 1, 1, 1, 0]
    bo = bop[i] + (bop[i + 1] - bop[i]) * f; to = top[i] + (top[i + 1] - top[i]) * f
    amt = [0, 0, 1, 1, 1, 1, 1, 1, 0]; a = amt[i] + (amt[i + 1] - amt[i]) * f
    c1 = L1.mean(0)
    if bo > 0.02:
        col = tuple(int(10 + (v - 10) * bo) for v in (167, 139, 250))
        for b in range(B.N_BANDS):
            sel = labels == b
            if not sel.any(): continue
            off = a * B.DRIFT * (c1 - np.c_[xs[sel], ys[sel]].mean(0))
            for x, y in zip(xs[sel] + off[0], ys[sel] + off[1]):
                d.rectangle([x * S, y * S, x * S + S - 1, y * S + S - 1], fill=col)
    if to > 0.02:
        stops = [P, P, L1, L1, L2, L2, L3, L3, P]
        pos = stops[i] + (stops[i + 1] - stops[i]) * f
        col = tuple(int(10 + (v - 10) * to) for v in (167, 139, 250))
        for x, y in pos:
            d.ellipse([x * S - 1.6, y * S - 1.6, x * S + 1.6, y * S + 1.6], fill=col)
    return img

ts = [5.3, 8.6, 11.9]
frames = [frame(t) for t in ts]
w, h = frames[0].size
sheet = Image.new("RGB", (w * 3, h))
for k, fr in enumerate(frames):
    ImageDraw.Draw(fr).text((6, 6), f"t={ts[k]}s", fill=(34, 211, 238))
    sheet.paste(fr, ((k % 3) * w, (k // 3) * h))
sheet.save(B.ROOT / "gen" / "preview.png")
print("ok", sheet.size)
