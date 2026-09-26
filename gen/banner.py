"""Generate dark.svg / light.svg profile banners.

Source of truth: this script + gen/data/*.npy. Never hand-edit the SVGs.
Usage: python gen/banner.py <source-image>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "gen" / "data"
DATA.mkdir(parents=True, exist_ok=True)

W, H = 1180, 610
GW, GH = 300, 340          # dither grid
CELL = 1.2                  # px per grid cell
SEED = 7
TONE_FLOOR, TONE_CEIL = 0.18, 0.92  # dot density range inside the subject

THEMES = {
    "dark": dict(bg="#0A101F", window="#0D1528", border="#1E293B", bar="#111a30",
                 frame="#0A101F", text="#E2E8F0", dim="#475569", chrome="#22D3EE",
                 portrait="#A78BFA", accent="#10B981", live="#EF4444", pilltext="#0A101F"),
    "light": dict(bg="#F8FAFC", window="#FFFFFF", border="#CBD5E1", bar="#EEF2F7",
                  frame="#F8FAFC", text="#0F172A", dim="#94A3B8", chrome="#0891B2",
                  portrait="#7C3AED", accent="#059669", live="#DC2626", pilltext="#FFFFFF"),
}

HANDLE = "@BSVS777"
ROWS = [
    [("Subject", "Bernal Vargas Solís"),
     ("Role", "Full-Stack Developer"),
     ("Origin", "Costa Rica"),
     ("Education", "Software Engineering @ UTN"),
     ("Status", "Building + Learning + Shipping"),
     ("ToolChain", "VS Code, Git, Docker, LazyVim")],
    [("Core.Lang", "Java, TypeScript, PHP, Python, C#"),
     ("Core.Frontend", "Svelte, Tailwind, HTML/CSS"),
     ("Core.Backend", "Laravel (TALL), Node.js"),
     ("Core.Database", "PostgreSQL, MySQL"),
     ("Core.Infra", "Docker, GitHub Actions")],
    [("Grid.Mail", "coming soon"),
     ("Grid.Portfolio", "coming soon"),
     ("Grid.LinkedIn", "in/bernal-vargas-solís"),
     ("Grid.GitHub", "github.com/BSVS777"),
     ("Grid.Facebook", "coming soon")],
]

# ---------------------------------------------------------------- portrait


def subject_mask(rgb: np.ndarray) -> np.ndarray:
    """Background segmentation: colour distance from the border colour."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    bgc = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(float) - bgc, axis=2)
    m = dist > 60
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:  # keep every large component (a logo can have several)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.05 * sizes.max()]
        m = np.isin(lab, keep)
    return m


def prepare(src: Path):
    im = Image.open(src).convert("RGB")
    rgb = np.asarray(im)
    mask = subject_mask(rgb)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.06 * max(x1 - x0, y1 - y0))
    box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
    crop = im.crop(box)
    mcrop = Image.fromarray(mask.astype(np.uint8) * 255).crop(box)
    # fit into the grid, centred, background black
    scale = min(GW / crop.width, GH / crop.height)
    size = (round(crop.width * scale), round(crop.height * scale))
    off = ((GW - size[0]) // 2, (GH - size[1]) // 2)
    canvas = Image.new("RGB", (GW, GH))
    canvas.paste(crop.resize(size, Image.LANCZOS), off)
    mcanvas = Image.new("L", (GW, GH))
    mcanvas.paste(mcrop.resize(size, Image.LANCZOS), off)
    # tone = brightest channel (the source is saturated blue; luma would crush it)
    tone = Image.fromarray(np.asarray(canvas).max(axis=2).astype(np.uint8))
    tone = ImageEnhance.Contrast(tone).enhance(1.3)
    tone = ImageOps.autocontrast(tone, cutoff=1)
    tone = tone.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    t = np.asarray(tone, float) / 255.0
    mask = np.asarray(mcanvas) > 127
    # stretch using subject pixels only; a black background would otherwise
    # push the whole subject to white and the dither turns solid
    lo, hi = np.percentile(t[mask], [2, 98])
    t = np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)
    t = TONE_FLOOR + (TONE_CEIL - TONE_FLOOR) * t
    return t, mask


def dither(tone: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1-bit Floyd-Steinberg, serpentine order. Returns bool grid of dots."""
    a = np.where(mask, tone, 0.0).copy()
    h, w = a.shape
    out = np.zeros_like(a, bool)
    for y in range(h):
        rng = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in rng:
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            e = old - new
            if 0 <= x + d < w:
                a[y, x + d] += e * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    a[y + 1, x - d] += e * 3 / 16
                a[y + 1, x] += e * 5 / 16
                if 0 <= x + d < w:
                    a[y + 1, x + d] += e * 1 / 16
    # hard-clear error-diffusion bleed outside the subject
    return out & mask


# ---------------------------------------------------------------- animation

N_GROUPS = 60
INTRO_SPREAD = 2.4  # s over which group starts are spread
INTRO_FADE = 0.8    # s per group fade


def intro_groups(dots: np.ndarray):
    rng = np.random.default_rng(SEED)
    ys, xs = np.nonzero(dots)
    g = rng.integers(0, N_GROUPS, size=len(xs))
    begins = np.sort(rng.uniform(0, INTRO_SPREAD, N_GROUPS))
    return xs, ys, g, begins


def evenness(xs, ys, g, begins, t=1.2, tiles=8):
    """Coefficient of variation of the revealed fraction across tiles at time t."""
    revealed = np.clip((t - begins[g]) / INTRO_FADE, 0, 1)
    tx, ty = xs * tiles // GW, ys * tiles // GH
    fr = []
    for i in range(tiles):
        for j in range(tiles):
            sel = (tx == i) & (ty == j)
            if sel.sum() >= 30:
                fr.append(revealed[sel].mean())
    fr = np.array(fr)
    return float(fr.std() / fr.mean())


def runs_path(xs, ys) -> str:
    """Merge horizontal neighbours into runs, emit compact path data."""
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    parts, i, n = [], 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and ys[j + 1] == ys[i] and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(f"M{xs[i]} {ys[i]}h{j - i + 1}v1h-{j - i + 1}z")
        i = j + 1
    return "".join(parts)


# ---------------------------------------------------------------- loop

LOGO_BOX = 250        # grid cells a logo may occupy (square)
N_TRAVEL = 1300
N_BANDS = 94
DRIFT = 0.42
BAND_NOISE = 4.0      # per-dot noise before grouping; avoids the square-grid trap
T_PORTRAIT, T_LOGO, T_MOVE = 3.0, 2.0, 1.3
LOOP = T_PORTRAIT + 3 * T_LOGO + 4 * T_MOVE
INTRO_END = INTRO_SPREAD + INTRO_FADE


def loop_key_times() -> str:
    t, ks = 0.0, [0.0]
    for step in (T_PORTRAIT, T_MOVE, T_LOGO, T_MOVE, T_LOGO, T_MOVE, T_LOGO, T_MOVE):
        t += step
        ks.append(t / LOOP)
    return ";".join(f"{k:.4f}" for k in ks)


KEY_SPLINES = ";".join(["0.45 0 0.2 1"] * 8)


def repair_ts(m: np.ndarray) -> np.ndarray:
    """The reference sheet's 'S' is half opaque; re-cut it as a bold S matched to the T."""
    from PIL import ImageDraw, ImageFont
    filled = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(filled & ~m)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    t_lab = 1 + int(np.argmax(sizes))
    ys, xs = np.nonzero(lab == t_lab)
    ty0, ty1, tx1 = ys.min(), ys.max(), xs.max()
    m = m | (filled & (lab != t_lab))            # drop the broken S remnants
    cap = ty1 - ty0 + 1
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", int(cap * 1.38))
    glyph = Image.new("L", m.shape[::-1])
    d = ImageDraw.Draw(glyph)
    bx0, by0, bx1, by1 = d.textbbox((0, 0), "S", font=font)
    d.text((tx1 + cap * 0.10 - bx0, ty0 - by0 + (cap - (by1 - by0)) / 2), "S", font=font, fill=255)
    return m & ~(np.asarray(glyph) > 127)


def load_logos(path: Path) -> list[np.ndarray]:
    """Split the reference sheet into logos by empty columns; skip the first (the 25)."""
    rgba = np.asarray(Image.open(path).convert("RGBA")).astype(int)
    # the sheet is transparent where the logos are "white": alpha is the only
    # reliable signal, the RGB under transparent pixels is leftover noise
    blue = (rgba[..., 3] > 128) & ((rgba[..., 2] - rgba[..., 0]) > 40)
    cols = np.nonzero(blue.any(axis=0))[0]
    cuts = np.nonzero(np.diff(cols) > 10)[0]
    starts = np.r_[cols[0], cols[cuts + 1]]
    ends = np.r_[cols[cuts], cols[-1]]
    masks = []
    for k, (x0, x1) in enumerate(list(zip(starts, ends))[-3:]):
        m = blue[:, x0:x1 + 1].copy()
        if k == 2:
            m = repair_ts(m)
        ys = np.nonzero(m.any(axis=1))[0]
        m = m[ys[0]:ys[-1] + 1]
        s = LOGO_BOX / max(m.shape)
        size = (max(1, round(m.shape[1] * s)), max(1, round(m.shape[0] * s)))
        small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.LANCZOS)) > 127
        grid = np.zeros((GH, GW), bool)
        ox, oy = (GW - size[0]) // 2, (GH - size[1]) // 2
        grid[oy:oy + size[1], ox:ox + size[0]] = small
        masks.append(grid)
    return masks


def _poisson(cands: np.ndarray, r: float) -> np.ndarray:
    """Greedy blue-noise: accept candidates in order if no accepted point within r."""
    blocked = np.zeros((GH, GW), bool)
    ri = int(np.ceil(r))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disk = (xx ** 2 + yy ** 2) < r * r
    keep = []
    for y, x in cands:
        if blocked[y, x]:
            continue
        keep.append((x, y))
        y0, y1, x0, x1 = max(0, y - ri), min(GH, y + ri + 1), max(0, x - ri), min(GW, x + ri + 1)
        blocked[y0:y1, x0:x1] |= disk[y0 - y + ri:y1 - y + ri, x0 - x + ri:x1 - x + ri]
    return np.array(keep, float) + 0.5


def sample_points(mask: np.ndarray, n: int, rng) -> np.ndarray:
    """Exactly n evenly spaced points; outline (incl. holes) first so glyphs stay legible."""
    edge = mask & ~ndimage.binary_erosion(mask, iterations=1)
    e = np.argwhere(edge)
    i = np.argwhere(mask & ~edge)
    cands = np.r_[e[rng.permutation(len(e))], i[rng.permutation(len(i))]]
    lo, hi = 1.0, 20.0
    for _ in range(25):  # bisect the spacing radius to land just above n points
        r = (lo + hi) / 2
        if len(_poisson(cands, r)) >= n:
            lo = r
        else:
            hi = r
    pts = _poisson(cands, lo)
    return pts[rng.choice(len(pts), n, replace=False)]


def match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Optimal transport (assignment): reorder dst so dst[i] is src[i]'s target."""
    from scipy.optimize import linear_sum_assignment
    cost = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
    _, cols = linear_sum_assignment(cost)
    return dst[cols]


def drift_bands(xs, ys, rng, noise=BAND_NOISE):
    from scipy.cluster.vq import kmeans2
    pts = np.c_[xs, ys].astype(float)
    noisy = pts + rng.normal(0, noise, pts.shape) if noise else pts
    _, labels = kmeans2(noisy, N_BANDS, seed=SEED, minit="++")
    return labels


def straight_boundary(xs, ys, labels) -> float:
    """Share of band-boundary cells that continue straight for 3 cells (0.01 organic, 0.17 grid)."""
    lab = np.full((GH, GW), -1)
    lab[ys, xs] = labels
    hits = total = 0
    for a in (lab, lab.T):
        both = (a[:, :-1] >= 0) & (a[:, 1:] >= 0)
        b = both & (a[:, :-1] != a[:, 1:])
        total += b.sum()
        hits += (b[1:-1] & b[:-2] & b[2:]).sum()
    return float(hits / max(total, 1))


# ---------------------------------------------------------------- svg

FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
ROW_CHARS = 72


def info_row(x, y, width, label, value, t):
    lead = ROW_CHARS - len(label) - len(value) - 2
    dots = "." * max(3, lead)
    return (f'<text x="{x}" y="{y}" font-size="14" textLength="{width}" '
            f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
            f'<tspan fill="{t["chrome"]}">{label}</tspan>'
            f'<tspan fill="{t["dim"]}"> {dots} </tspan>'
            f'<tspan fill="{t["text"]}">{value}</tspan></text>')


def build_svg(theme: str, xs, ys, g, begins, loop) -> str:
    t = THEMES[theme]
    fx, fy, fw, fh = 34, 62, 424, 524
    px = fx + (fw - GW * CELL) / 2
    py = fy + 34 + (fh - 34 - GH * CELL) / 2

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
           f'<rect x="10" y="10" width="{W - 20}" height="{H - 20}" rx="12" '
           f'fill="{t["window"]}" stroke="{t["border"]}"/>',
           f'<path d="M10 22a12 12 0 0 1 12-12h{W - 44}a12 12 0 0 1 12 12v24h-{W - 20}z" '
           f'fill="{t["bar"]}"/>',
           f'<line x1="10" y1="46" x2="{W - 10}" y2="46" stroke="{t["border"]}"/>']
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        out.append(f'<circle cx="{32 + i * 20}" cy="28" r="6" fill="{c}"/>')
    out.append(f'<text x="{W / 2}" y="33" font-size="13" text-anchor="middle" '
               f'fill="{t["dim"]}">profile.sh --live</text>')

    # portrait frame
    out.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="8" '
               f'fill="{t["frame"]}" stroke="{t["border"]}"/>')
    out.append(f'<text x="{fx + 16}" y="{fy + 24}" font-size="13" fill="{t["chrome"]}" '
               f'textLength="92" lengthAdjust="spacingAndGlyphs">VISUAL.MAP</text>')
    out.append(f'<text x="{fx + fw - 16}" y="{fy + 24}" font-size="12" text-anchor="end" '
               f'fill="{t["dim"]}">{GW}x{GH} · 1-bit</text>')
    out.append(f'<g transform="translate({px:.1f} {py:.1f}) scale({CELL})" '
               f'fill="{t["portrait"]}" shape-rendering="crispEdges">')
    # intro layer: plays once, then hands over to the loop layer
    out.append('<g>')
    for k in np.argsort(begins):
        sel = g == k
        out.append(f'<path opacity="0" d="{runs_path(xs[sel], ys[sel])}">'
                   f'<animate attributeName="opacity" from="0" to="1" '
                   f'begin="{begins[k]:.2f}s" dur="{INTRO_FADE}s" fill="freeze"/></path>')
    out.append(f'<set attributeName="visibility" to="hidden" begin="{INTRO_END}s" fill="freeze"/></g>')

    # loop layer 1: portrait drift bands
    timing = (f'dur="{LOOP}s" begin="{INTRO_END}s" repeatCount="indefinite" '
              f'keyTimes="{loop_key_times()}"')
    spline = f'calcMode="spline" keySplines="{KEY_SPLINES}"'
    out.append(f'<g visibility="hidden"><set attributeName="visibility" to="visible" '
               f'begin="{INTRO_END}s" fill="freeze"/>')
    labels, drifts = loop["labels"], loop["drifts"]
    for b in range(N_BANDS):
        sel = labels == b
        if not sel.any():
            continue
        dx, dy = drifts[b]
        d = f"{dx:.1f} {dy:.1f}"
        out.append(f'<path d="{runs_path(xs[sel], ys[sel])}">'
                   f'<animateTransform attributeName="transform" type="translate" '
                   f'values="0 0;0 0;{";".join([d] * 6)};0 0" {timing} {spline}/>'
                   f'<animate attributeName="opacity" values="1;1;0;0;0;0;0;0;1" {timing}/></path>')
    out.append('</g>')

    # loop layer 2: travellers morphing between logos
    out.append(f'<g opacity="0" shape-rendering="auto"><animate attributeName="opacity" '
               f'values="0;0;1;1;1;1;1;1;0" {timing}/>')
    P, L1, L2, L3 = loop["paths"]
    for i in range(len(P)):
        stops = [P[i], P[i], L1[i], L1[i], L2[i], L2[i], L3[i], L3[i], P[i]]
        vals = ";".join(f"{x:.1f} {y:.1f}" for x, y in stops)
        out.append(f'<rect x="-0.8" y="-0.8" width="1.6" height="1.6">'
                   f'<animateTransform attributeName="transform" type="translate" '
                   f'values="{vals}" {timing} {spline}/></rect>')
    out.append('</g></g>')

    # info panel
    ix, iw = 492, 650
    out.append(f'<text x="{ix}" y="{fy + 24}" font-size="13" fill="{t["chrome"]}" '
               f'textLength="100" lengthAdjust="spacingAndGlyphs">SYSTEM.INFO</text>')
    lx = ix + 118
    out.append(f'<rect x="{lx}" y="{fy + 10}" width="58" height="20" rx="10" fill="none" '
               f'stroke="{t["live"]}"/>')
    out.append(f'<circle cx="{lx + 13}" cy="{fy + 20}" r="4" fill="{t["live"]}">'
               f'<animate attributeName="opacity" values="1;0.2;1" dur="1.6s" '
               f'repeatCount="indefinite"/></circle>')
    out.append(f'<text x="{lx + 23}" y="{fy + 24.5}" font-size="12" fill="{t["live"]}" '
               f'textLength="28" lengthAdjust="spacingAndGlyphs">LIVE</text>')
    pw = 108
    out.append(f'<rect x="{ix + iw - pw}" y="{fy + 7}" width="{pw}" height="26" rx="13" '
               f'fill="{t["accent"]}"/>')
    out.append(f'<text x="{ix + iw - pw / 2}" y="{fy + 25}" font-size="14" text-anchor="middle" '
               f'fill="{t["pilltext"]}" font-weight="700" textLength="{pw - 24}" '
               f'lengthAdjust="spacingAndGlyphs">{HANDLE}</text>')
    out.append(f'<line x1="{ix}" y1="{fy + 44}" x2="{ix + iw}" y2="{fy + 44}" '
               f'stroke="{t["border"]}" stroke-dasharray="2 4"/>')
    y = fy + 72
    for gi, group in enumerate(ROWS):
        for label, value in group:
            out.append(info_row(ix, y, iw, label, value, t))
            y += 23
        if gi < len(ROWS) - 1:
            out.append(f'<line x1="{ix}" y1="{y - 9}" x2="{ix + iw}" y2="{y - 9}" '
                       f'stroke="{t["border"]}" stroke-dasharray="2 4"/>')
            y += 12
    out.append('</svg>')
    return "\n".join(out)


def main():
    src = Path(sys.argv[1])
    tone, mask = prepare(src)
    dots = dither(tone, mask)
    np.save(DATA / "tone.npy", tone)
    np.save(DATA / "mask.npy", mask)
    np.save(DATA / "dots.npy", dots)
    xs, ys, g, begins = intro_groups(dots)
    print(f"dots={dots.sum()}  coverage={dots.sum() / mask.sum():.2f}  "
          f"evenness@1.2s={evenness(xs, ys, g, begins):.3f}")

    rng = np.random.default_rng(SEED)
    logos = load_logos(ROOT / "gen" / "logos.webp")
    P = sample_points(mask, N_TRAVEL, rng)
    L1 = match(P, sample_points(logos[0], N_TRAVEL, rng))
    L2 = match(L1, sample_points(logos[1], N_TRAVEL, rng))
    L3 = match(L2, sample_points(logos[2], N_TRAVEL, rng))
    np.save(DATA / "travellers.npy", np.stack([P, L1, L2, L3]))
    labels = drift_bands(xs, ys, rng)
    np.save(DATA / "bands.npy", labels)
    c1 = L1.mean(axis=0)
    drifts = np.zeros((N_BANDS, 2))
    for b in range(N_BANDS):
        sel = labels == b
        if sel.any():
            drifts[b] = DRIFT * (c1 - np.c_[xs[sel], ys[sel]].mean(axis=0))
    grid_labels = drift_bands(xs, ys, np.random.default_rng(SEED), noise=0)
    print(f"straight-boundary: noisy={straight_boundary(xs, ys, labels):.3f}  "
          f"no-noise={straight_boundary(xs, ys, grid_labels):.3f}  loop={LOOP:.1f}s")
    loop = dict(labels=labels, drifts=drifts, paths=(P, L1, L2, L3))

    for theme in THEMES:
        svg = build_svg(theme, xs, ys, g, begins, loop)
        (ROOT / f"{theme}.svg").write_text(svg, encoding="utf-8")
        print(f"{theme}.svg {len(svg.encode()) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
