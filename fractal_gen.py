#!/usr/bin/env python3
"""
fractgen.py - Batch generator for fractal & strange-attractor wallpapers.

Generates large batches of high-resolution images with randomized parameters.
Two families, both effectively infinite in variety:

  julia     - Smooth-colored Julia sets. Variety comes from the complex
              constant c (sampled near the Mandelbrot boundary, where the
              interesting structure lives).
  attractor - Clifford / De Jong strange attractors, rendered as log-density
              fields. Four random params each -> wildly different every time.

Usage examples:
  python fractgen.py --count 2000 --type both --width 3840 --height 2160 --out wallpapers
  python fractgen.py --count 500 --type attractor --width 2560 --height 1440
  python fractgen.py --count 50 --type julia --seed 42        # reproducible batch

Tuning knobs that matter:
  --iters       Julia max iterations (detail near the boundary). 256-512 is plenty.
  --points      Attractor sample count in millions. More = smoother, slower. 12 is good.
  --workers     Parallel processes. Defaults to CPU count.
"""

import argparse
import os
import random
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
from numba import njit, prange
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ----------------------------------------------------------------------------
# Palettes: a mix of perceptually-nice matplotlib maps + a few custom gradients.
# ----------------------------------------------------------------------------
_MPL_MAPS = [
    "magma", "inferno", "viridis", "plasma", "cividis", "twilight",
    "twilight_shifted", "cubehelix", "turbo", "gnuplot2", "CMRmap",
    "ocean", "gist_earth", "terrain", "nipy_spectral",
]

_CUSTOM = {
    "ember":   ["#000000", "#1a0033", "#7a0c2e", "#e3611c", "#ffd16a", "#fffbe6"],
    "abyss":   ["#000814", "#001d3d", "#003566", "#0077b6", "#48cae4", "#caf0f8"],
    "neon":    ["#03001e", "#7303c0", "#ec38bc", "#fdeff9"],
    "gold":    ["#000000", "#241105", "#7a4a17", "#d9a441", "#fff1c1"],
    "forest":  ["#011a0e", "#0b3d2e", "#1f7a4d", "#7ed957", "#eafcd2"],
    "ice":     ["#000a14", "#0b2545", "#3a6ea5", "#8bd3e6", "#f0f7ff"],
}


def random_cmap(rng):
    """Return a matplotlib colormap, sometimes reversed, from the pooled set."""
    if rng.random() < 0.45:
        name, colors = rng.choice(list(_CUSTOM.items()))
        cmap = LinearSegmentedColormap.from_list(name, colors)
    else:
        cmap = matplotlib.colormaps[rng.choice(_MPL_MAPS)]
    if rng.random() < 0.5:
        cmap = cmap.reversed()
    return cmap


# ----------------------------------------------------------------------------
# JULIA SETS
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def _julia_kernel(w, h, cx, cy, x0, x1, y0, y1, max_iter, escape2):
    out = np.zeros((h, w), dtype=np.float64)
    dx = (x1 - x0) / w
    dy = (y1 - y0) / h
    log2 = np.log(2.0)
    for j in prange(h):
        zy0 = y0 + j * dy
        for i in range(w):
            zx = x0 + i * dx
            zy = zy0
            n = 0
            zx2 = zx * zx
            zy2 = zy * zy
            while zx2 + zy2 <= escape2 and n < max_iter:
                zy = 2.0 * zx * zy + cy
                zx = zx2 - zy2 + cx
                zx2 = zx * zx
                zy2 = zy * zy
                n += 1
            if n >= max_iter:
                out[j, i] = 0.0  # inside the set -> deepest color
            else:
                # smooth (continuous) iteration count
                mag = zx2 + zy2
                nu = np.log(np.log(mag) / 2.0 / log2) / log2
                out[j, i] = n + 1.0 - nu
    return out


def _interesting_c(rng):
    """Pick a Julia constant that yields rich structure."""
    mode = rng.random()
    if mode < 0.5:
        # rotating family on the dendrite radius - reliably gorgeous
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.7, 0.7995)
        return r * np.cos(theta), r * np.sin(theta)
    elif mode < 0.8:
        # near well-known seed points
        seeds = [(-0.8, 0.156), (-0.4, 0.6), (0.285, 0.01),
                 (-0.70176, -0.3842), (-0.835, -0.2321), (0.45, 0.1428)]
        bx, by = rng.choice(seeds)
        return bx + rng.uniform(-0.03, 0.03), by + rng.uniform(-0.03, 0.03)
    else:
        # broad random near the boundary
        return rng.uniform(-0.9, 0.4), rng.uniform(-0.8, 0.8)


def render_julia(w, h, seed):
    rng = random.Random(seed)
    cx, cy = _interesting_c(rng)
    max_iter = rng.choice([256, 384, 512])
    # frame the set; add a little random zoom/pan for variety
    zoom = rng.uniform(1.1, 1.7)
    aspect = w / h
    span_y = 2.6 / zoom
    span_x = span_y * aspect
    panx = rng.uniform(-0.25, 0.25)
    pany = rng.uniform(-0.25, 0.25)
    x0, x1 = -span_x / 2 + panx, span_x / 2 + panx
    y0, y1 = -span_y / 2 + pany, span_y / 2 + pany

    field = _julia_kernel(w, h, cx, cy, x0, x1, y0, y1, max_iter, 256.0)

    # Histogram-equalize the escape field: rank each pixel by its value so color
    # spreads across the actual structure instead of pooling in a flat exterior.
    flat = field.ravel()
    order = flat.argsort()
    ranks = np.empty(flat.shape[0], dtype=np.float64)
    ranks[order] = np.arange(flat.shape[0])
    eq = (ranks / (flat.shape[0] - 1)).reshape(field.shape)
    eq = np.power(eq, rng.uniform(0.8, 1.1))
    cmap = random_cmap(rng)
    rgb = (cmap(eq)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


# ----------------------------------------------------------------------------
# STRANGE ATTRACTORS (Clifford & De Jong)
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _lyapunov(a, b, c, d, kind):
    """Largest Lyapunov exponent via the Jacobian-stretched tangent vector.
    Positive -> chaotic (space-filling strange attractor). <=0 -> point/cycle."""
    x = 0.1; y = 0.1
    vx = 1.0; vy = 0.0
    lsum = 0.0
    n = 0
    for it in range(2000):
        if kind == 0:  # Clifford
            nx = np.sin(a * y) + c * np.cos(a * x)
            ny = np.sin(b * x) + d * np.cos(b * y)
            j00 = -c * a * np.sin(a * x); j01 = a * np.cos(a * y)
            j10 = b * np.cos(b * x);      j11 = -d * b * np.sin(b * y)
        else:          # De Jong
            nx = np.sin(a * y) - np.cos(b * x)
            ny = np.sin(c * x) - np.cos(d * y)
            j00 = b * np.sin(b * x); j01 = a * np.cos(a * y)
            j10 = c * np.cos(c * x); j11 = d * np.sin(d * y)
        x = nx; y = ny
        # propagate tangent vector
        tx = j00 * vx + j01 * vy
        ty = j10 * vx + j11 * vy
        norm = np.sqrt(tx * tx + ty * ty)
        if norm < 1e-12:
            return -1.0
        if it > 100:  # discard transient
            lsum += np.log(norm)
            n += 1
        vx = tx / norm
        vy = ty / norm
    if n == 0:
        return -1.0
    return lsum / n


@njit(cache=True, fastmath=True)
def _attractor_bounds(a, b, c, d, n, kind):
    x = 0.0; y = 0.0
    minx = 1e9; maxx = -1e9; miny = 1e9; maxy = -1e9
    for _ in range(n):
        if kind == 0:  # Clifford
            nx = np.sin(a * y) + c * np.cos(a * x)
            ny = np.sin(b * x) + d * np.cos(b * y)
        else:          # De Jong
            nx = np.sin(a * y) - np.cos(b * x)
            ny = np.sin(c * x) - np.cos(d * y)
        x = nx; y = ny
        if x < minx: minx = x
        if x > maxx: maxx = x
        if y < miny: miny = y
        if y > maxy: maxy = y
    return minx, maxx, miny, maxy


@njit(cache=True, fastmath=True)
def _attractor_density(a, b, c, d, n, kind, w, h, minx, maxx, miny, maxy):
    grid = np.zeros((h, w), dtype=np.float64)
    rangex = maxx - minx
    rangey = maxy - miny
    if rangex <= 0: rangex = 1e-6
    if rangey <= 0: rangey = 1e-6
    sx = (w - 1) / rangex
    sy = (h - 1) / rangey
    x = 0.0; y = 0.0
    for _ in range(n):
        if kind == 0:
            nx = np.sin(a * y) + c * np.cos(a * x)
            ny = np.sin(b * x) + d * np.cos(b * y)
        else:
            nx = np.sin(a * y) - np.cos(b * x)
            ny = np.sin(c * x) - np.cos(d * y)
        x = nx; y = ny
        px = int((x - minx) * sx)
        py = int((y - miny) * sy)
        if 0 <= px < w and 0 <= py < h:
            grid[py, px] += 1.0
    return grid


def render_attractor(w, h, seed):
    rng = random.Random(seed)
    n = render_attractor.points

    # Resample params until we find a genuinely chaotic (space-filling) attractor.
    kind = a = b = c = d = None
    for _ in range(400):
        kind = rng.randint(0, 1)
        a = rng.uniform(-3, 3); b = rng.uniform(-3, 3)
        c = rng.uniform(-3, 3); d = rng.uniform(-3, 3)
        if _lyapunov(a, b, c, d, kind) > 0.05:   # positive => chaotic
            break

    minx, maxx, miny, maxy = _attractor_bounds(a, b, c, d, min(n, 400_000), kind)
    # pad bounds slightly so we don't clip
    px = (maxx - minx) * 0.02 + 1e-3
    py = (maxy - miny) * 0.02 + 1e-3
    minx -= px; maxx += px; miny -= py; maxy += py

    # keep aspect ratio of the attractor inside the frame (letterbox the data range)
    data_aspect = (maxx - minx) / (maxy - miny)
    target_aspect = w / h
    if data_aspect > target_aspect:
        # too wide -> expand y range
        cy = (miny + maxy) / 2
        new_h = (maxx - minx) / target_aspect
        miny, maxy = cy - new_h / 2, cy + new_h / 2
    else:
        cx = (minx + maxx) / 2
        new_w = (maxy - miny) * target_aspect
        minx, maxx = cx - new_w / 2, cx + new_w / 2

    grid = _attractor_density(a, b, c, d, n, kind, w, h, minx, maxx, miny, maxy)
    grid = np.log1p(grid)
    m = grid.max()
    if m <= 0:
        # degenerate params -> retry with a fresh seed
        return render_attractor(w, h, seed + 999983)
    grid = grid / m
    grid = np.power(grid, rng.uniform(0.7, 1.0))
    cmap = random_cmap(rng)
    rgb = (cmap(grid)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


render_attractor.points = 12_000_000  # set from CLI


# ----------------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------------
def _worker(task):
    idx, kind, w, h, seed, outdir = task
    try:
        if kind == "julia":
            img = render_julia(w, h, seed)
        else:
            img = render_attractor(w, h, seed)
        path = os.path.join(outdir, f"{kind}_{idx:06d}.png")
        img.save(path, optimize=True)
        return path
    except Exception as e:  # never let one bad param set kill the batch
        return f"ERR {idx}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Batch fractal / attractor wallpaper generator")
    ap.add_argument("--count", type=int, default=100, help="how many images")
    ap.add_argument("--type", choices=["julia", "attractor", "both"], default="both")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=None, help="override Julia max iterations")
    ap.add_argument("--points", type=int, default=12, help="attractor sample points, millions")
    ap.add_argument("--out", default="wallpapers")
    ap.add_argument("--seed", type=int, default=None, help="base seed for reproducibility")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    render_attractor.points = args.points * 1_000_000

    base = args.seed if args.seed is not None else random.randrange(1 << 30)
    workers = args.workers or cpu_count()

    tasks = []
    for i in range(args.count):
        if args.type == "both":
            kind = "julia" if i % 2 == 0 else "attractor"
        else:
            kind = args.type
        tasks.append((i, kind, args.width, args.height, base + i * 7919, args.out))

    print(f"Generating {args.count} images ({args.width}x{args.height}) "
          f"-> {args.out}/  using {workers} workers, base seed {base}")
    done = 0
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            done += 1
            if res.startswith("ERR"):
                print(res)
            if done % 25 == 0 or done == args.count:
                print(f"  {done}/{args.count}")
    print("Done.")


if __name__ == "__main__":
    main()
