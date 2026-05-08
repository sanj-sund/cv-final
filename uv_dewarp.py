"""
UV-based dewarping using a ground-truth UV map.

Given an RGB render of a warped document and the corresponding UV map
(where each pixel encodes the (u, v) coordinate on the flat paper that
projects to that screen pixel), produce a flat dewarped image by
scattering RGB pixels into UV space.

Usage:
    python uv_dewarp.py --rgb path/to/rgb.jpg --uv path/to/uv.png \
                       --out dewarped.png [--size 512] [--mask path/to/border.png]

If --rgb / --uv are omitted, the script picks the first matching pair
from renders/synthetic_data_pitch_sweep/{rgb,uv}/.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA_DIR = Path("renders/synthetic_data_pitch_sweep")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_uv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a UV map. Returns (uv[H,W,2] float in [0,1], fg_mask[H,W] bool).

    The dataset encodes (U, V, 0) in (R, G, B). Background pixels are a
    uniform gray (R == G == B) from the world clear color — anything where
    the channels differ is paper. Handles both 8-bit and 16-bit PNGs.
    """
    raw = np.asarray(Image.open(path))
    if raw.ndim == 2:
        raise ValueError(f"Expected RGB image, got grayscale: {path}")
    if raw.shape[-1] == 4:
        raw = raw[..., :3]
    denom = 65535.0 if raw.dtype == np.uint16 else 255.0
    arr = raw.astype(np.float32) / denom
    # Background is uniform gray (linear render of world color); paper has
    # distinct U / V values so its channels differ.
    bg = (raw[..., 0] == raw[..., 1]) & (raw[..., 1] == raw[..., 2])
    fg = ~bg
    return arr[..., :2], fg


def load_mask(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return m > 0.5


def dewarp_with_uv(
    rgb: np.ndarray,
    uv: np.ndarray,
    out_size: int = 512,
    mask: np.ndarray | None = None,
    flip_v: bool = True,
) -> np.ndarray:
    """
    Dewarp a warped document image using its ground-truth UV map.

    THE BIG IDEA
    ------------
    A UV map tells us, for every screen pixel (x, y) in the rendered image,
    which point (u, v) on the original flat paper that pixel came from.
    `u` and `v` are normalised paper coordinates in [0, 1]:
        (u=0, v=0) = one corner of the flat paper
        (u=1, v=1) = the opposite corner.

    To "dewarp" the document, we want to undo the camera + paper-bending
    transform and end up with an image of the flat paper. We do that by
    re-arranging pixels:

        screen pixel (x, y)  ── has colour rgb[y, x]
                             ── has UV     uv[y, x] = (u, v)
        place that colour onto the flat canvas at (u * W, v * H).

    This is called a *forward warp* (or "scatter"): we push each source
    pixel into the output. The opposite, *inverse warp* (for each output
    pixel, look up which source pixel it came from), would also work but
    needs interpolating the irregular UV samples — forward scatter is the
    most direct mapping that "does what the UV map literally tells you".

    BILINEAR SPLATTING (why each pixel paints 4 cells, not 1)
    ---------------------------------------------------------
    A source pixel rarely lands exactly on an integer canvas cell — its
    target (u·W, v·H) is fractional, e.g. (123.7, 88.3). If we just rounded
    to (124, 88) we'd lose information and create stair-stepping artefacts.
    Instead we distribute the colour over the 4 nearest cells with weights
    that depend on the fractional offset:

         (x0,y0) ── w00=(1-wx)(1-wy) ──┐
         (x1,y0) ── w01=wx*(1-wy)   ──┤  weights sum to 1.0
         (x0,y1) ── w10=(1-wx)*wy   ──┤
         (x1,y1) ── w11=wx*wy       ──┘

    Each cell accumulates a weighted sum of contributing colours and a
    running total of weights. At the end we divide colour-by-weight to get
    the final averaged value per cell. (This is exactly the math your GPU
    does behind the scenes when sampling textures with bilinear filtering.)

    HOLE FILL (why we have an iterative dilation pass)
    --------------------------------------------------
    Forward scatter has a known weakness: in regions of the source where
    paper geometry is "stretched" (foreshortened far edges of the paper,
    or thin slivers seen edge-on), few screen pixels map into a relatively
    large area of UV space, leaving canvas cells with no contributors. We
    fix these holes with a conservative dilation pass that ONLY paints
    empty cells, copying from filled neighbours — never overwriting a cell
    that already has data. That keeps it from smearing the document and
    creating the starburst-like artefacts you'd get from blurring.

    LIMITATIONS
    -----------
    1. Lighting / shadows from the rendered scene are baked into rgb[y, x]
       and follow it onto the flat canvas — the dewarp can fix geometry
       but cannot relight the document. Don't expect pixel-exact match
       against ground_truth.
    2. If the paper folds over itself, two different screen pixels can
       have the same UV coordinate. We just average them, which produces
       a faded ghost where the fold was — there's no clean answer here
       without a fold-aware model.
    3. We assume the UV map is *linear* (R=u, G=v, both in [0, 1]). If the
       UV PNG was rendered with Blender's default Filmic view transform,
       the values will be tone-mapped and you'll need to fix the renderer
       (set view_transform='Raw') before this function will produce the
       expected result.

    Args:
        rgb:    (H, W, 3) uint8 — source rendered image of the warped doc.
        uv:     (H, W, 2) float in [0, 1] — channel 0 = u (R), 1 = v (G).
        out_size: side length of the (square) flat output canvas in pixels.
        mask:   optional (H, W) bool — only splat pixels where True. Used
                to ignore background pixels whose UV is meaningless.
        flip_v: image y-axis points DOWN, but UV V=0 conventionally lands
                at the BOTTOM of the texture. Setting flip_v=True puts the
                document right-side-up in the output. Disable if your
                particular dataset uses the opposite convention.

    Returns:
        (out_size, out_size, 3) uint8 — the dewarped flat document image.
    """
    # ------------------------------------------------------------------
    # 0. Sanity check shapes
    # ------------------------------------------------------------------
    H, W, _ = rgb.shape
    assert uv.shape[:2] == (H, W), "rgb and uv must have matching HxW"

    # ------------------------------------------------------------------
    # 1. Extract U / V channels and (optionally) flip the V axis so the
    #    saved canvas reads top-to-bottom in screen orientation.
    # ------------------------------------------------------------------
    u = uv[..., 0]
    v = uv[..., 1]
    if flip_v:
        v = 1.0 - v

    # ------------------------------------------------------------------
    # 2. Build a "valid" boolean mask: which source pixels do we actually
    #    want to splat onto the canvas?
    #      - UV must be inside [0, 1] (out-of-range = not on the paper)
    #      - if a foreground mask was provided, the pixel must be on the
    #        document (not background).
    # ------------------------------------------------------------------
    valid = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    if mask is not None:
        valid &= mask

    # ------------------------------------------------------------------
    # 3. Convert each valid source pixel's UV to a *continuous* canvas
    #    position (fx, fy). E.g. u=0.5 with out_size=512 -> fx=255.5.
    # ------------------------------------------------------------------
    fx = u[valid] * (out_size - 1)
    fy = v[valid] * (out_size - 1)
    cols = rgb[valid].astype(np.float32)  # the colours we'll splat

    # ------------------------------------------------------------------
    # 4. For bilinear splatting, find the 4 integer canvas cells around
    #    each fractional position (x0,y0)–(x1,y1), and the fractional
    #    offsets (wx, wy) that determine how much weight each cell gets.
    # ------------------------------------------------------------------
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, out_size - 1)  # +1 cell to the right
    y1 = np.clip(y0 + 1, 0, out_size - 1)  # +1 cell down
    x0 = np.clip(x0, 0, out_size - 1)      # clamp inside canvas just in case
    y0 = np.clip(y0, 0, out_size - 1)

    wx = fx - x0  # how far right of x0 (in [0, 1))
    wy = fy - y0  # how far down  of y0 (in [0, 1))

    # The four bilinear weights — note they sum to 1 for every source pixel.
    w00 = (1 - wx) * (1 - wy)  # top-left cell
    w01 = wx       * (1 - wy)  # top-right cell
    w10 = (1 - wx) * wy        # bottom-left cell
    w11 = wx       * wy        # bottom-right cell

    # ------------------------------------------------------------------
    # 5. Accumulator buffers. `canvas` collects colour * weight; `weights`
    #    collects the matching total weights so we can normalise at the end.
    #    Many source pixels may land on the same cell — np.add.at handles
    #    repeated indices correctly (a normal canvas[ys, xs] += ... would
    #    only write the *last* contribution at each duplicated index).
    # ------------------------------------------------------------------
    canvas  = np.zeros((out_size, out_size, 3), dtype=np.float32)
    weights = np.zeros((out_size, out_size),    dtype=np.float32)

    for ys, xs, w in [(y0, x0, w00), (y0, x1, w01),
                      (y1, x0, w10), (y1, x1, w11)]:
        np.add.at(canvas,  (ys, xs), cols * w[:, None])  # weighted colour
        np.add.at(weights, (ys, xs), w)                  # weight bookkeeping

    # Normalise: each cell holds the average of all the source colours
    # that contributed to it (weighted by how close their target was).
    nz = weights > 0
    canvas[nz] /= weights[nz, None]

    # ------------------------------------------------------------------
    # 6. Hole-fill pass for cells that received zero contributions.
    #    Up to 8 iterations of an 8-neighbour dilation, but ONLY into
    #    empty cells. Filled cells are never overwritten, which prevents
    #    edge smearing and starburst-like artefacts you'd get from a
    #    naive blur.
    # ------------------------------------------------------------------
    holes = ~nz
    for _ in range(8):
        if not holes.any():
            break  # all holes are filled; we're done

        acc    = np.zeros_like(canvas)                                # neighbour colours
        wt     = np.zeros((out_size, out_size), dtype=np.float32)     # neighbour count
        filled = (~holes).astype(np.float32)                          # 1 where we have data

        # Sum contributions from each of the 8 neighbouring positions.
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            shifted   = np.roll(canvas, (dy, dx), axis=(0, 1))
            shifted_f = np.roll(filled, (dy, dx), axis=(0, 1))
            acc += shifted * shifted_f[..., None]   # colour from filled neighbours
            wt  += shifted_f                        # number of filled neighbours

        # Only write into holes that actually had at least one filled neighbour.
        fill = holes & (wt > 0)
        canvas[fill] = acc[fill] / wt[fill, None]
        holes = holes & ~fill  # the filled cells stop being holes

    # Final clamp + cast back to uint8 for saving as an 8-bit image.
    return np.clip(canvas, 0, 255).astype(np.uint8)


def find_first_pair(data_dir: Path) -> tuple[Path, Path]:
    rgb_dir = data_dir / "rgb"
    uv_dir = data_dir / "uv"
    for rgb_path in sorted(rgb_dir.glob("*.jpg")):
        uv_path = uv_dir / f"{rgb_path.stem}.png"
        if uv_path.exists():
            return rgb_path, uv_path
    raise FileNotFoundError(f"No matching rgb/uv pair found in {data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", type=Path, default=None)
    ap.add_argument("--uv", type=Path, default=None)
    ap.add_argument("--mask", type=Path, default=None, help="Optional border mask")
    ap.add_argument("--out", type=Path, default=Path("dewarped.png"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-flip-v", action="store_true",
                    help="Disable the default v-axis flip "
                         "(use if output ends up upside-down on a sample)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--show", action="store_true",
                    help="Open a matplotlib window with input vs output")
    ap.add_argument("--side-by-side", type=Path, default=None,
                    help="Also save a side-by-side PNG to this path")
    args = ap.parse_args()

    if args.rgb is None or args.uv is None:
        rgb_path, uv_path = find_first_pair(args.data_dir)
        print(f"Using auto-picked pair:\n  rgb = {rgb_path}\n  uv  = {uv_path}")
    else:
        rgb_path, uv_path = args.rgb, args.uv

    rgb = load_rgb(rgb_path)
    uv, fg_mask = load_uv(uv_path)
    mask = load_mask(args.mask) if args.mask else fg_mask

    if uv.shape[:2] != rgb.shape[:2]:
        # Resize UV (nearest) to match RGB so coordinates stay valid.
        # Reload so we keep the original dtype (8 / 16 bit) for the mask.
        raw = np.asarray(Image.open(uv_path))[..., :3]
        denom = 65535.0 if raw.dtype == np.uint16 else 255.0
        uv_arr_f = raw.astype(np.float32) / denom
        # Resize floats per-channel via PIL on a uint8 proxy is lossy; use
        # Image.NEAREST on the original to preserve exact codes for the mask
        # and bilinear on floats for the UV values.
        uv_resized = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
        for c in range(3):
            ch = Image.fromarray(uv_arr_f[..., c])  # PIL F mode
            uv_resized[..., c] = np.asarray(
                ch.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR)
            )
        uv = uv_resized[..., :2]
        if args.mask is None:
            bg = (uv_resized[..., 0] == uv_resized[..., 1]) & \
                 (uv_resized[..., 1] == uv_resized[..., 2])
            mask = ~bg

    flip_v = not args.no_flip_v

    # Dewarp the RGB image (the actual output we care about).
    out = dewarp_with_uv(rgb, uv, out_size=args.size, mask=mask, flip_v=flip_v)
    Image.fromarray(out).save(args.out)
    print(f"Wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")

    if args.side_by_side is not None or args.show:
        # Build a 5-panel illustration:
        #   [Input RGB] | [UV map] | [UV values dewarped] | [RGB dewarped] | [Ground truth]
        # - UV map column shows the colour-coded UV input (R=U, G=V).
        # - "UV values dewarped" splats the UV colours themselves through
        #   dewarp_with_uv — the result is the canonical rectangular
        #   gradient (black/red/green/yellow corners), which is visual proof
        #   that the geometric mapping is correct, independent of any image
        #   content.
        # Ground truth is loaded from the parallel `ground_truth/` directory
        # if present, otherwise the GT panel is omitted.
        from PIL import ImageDraw, ImageFont

        # 1) Encode UV as displayable RGB (uint8, B = 0).
        uv_disp_f = np.zeros((*uv.shape[:2], 3), dtype=np.float32)
        uv_disp_f[..., :2] = uv
        uv_rgb = (np.clip(uv_disp_f, 0, 1) * 255).astype(np.uint8)

        # 2) Dewarp the UV-as-image through the same pipeline.
        uv_dewarped = dewarp_with_uv(uv_rgb, uv, out_size=args.size,
                                     mask=mask, flip_v=flip_v)

        # 3) Resize all panels to the output canvas height.
        h = out.shape[0]

        def _resize_to_h(arr: np.ndarray) -> np.ndarray:
            pil = Image.fromarray(arr)
            new_w = int(round(pil.width * (h / pil.height)))
            return np.asarray(pil.resize((new_w, h), Image.BILINEAR))

        columns = [
            (_resize_to_h(rgb),    "Input (warped)"),
            (_resize_to_h(uv_rgb), "UV map (R=U, G=V)"),
            (uv_dewarped,          "UV values dewarped"),
            (out,                  "RGB dewarped"),
        ]

        # 4) Optional ground-truth column — looks for ground_truth/<stem>.png
        #    parallel to uv/<stem>.png.
        gt_path = uv_path.parent.parent / "ground_truth" / f"{uv_path.stem}.png"
        if gt_path.exists():
            gt_img = np.asarray(
                Image.open(gt_path).convert("RGB").resize((args.size, h), Image.BILINEAR)
            )
            columns.append((gt_img, "Ground truth"))

        # 5) Concatenate the image row.
        panel = np.concatenate([col for col, _ in columns], axis=1)

        # 6) Add a label strip beneath each column.
        font_size = 28
        label_h = font_size + 24  # padding above + below the glyph row
        strip = Image.new("RGB", (panel.shape[1], label_h), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        font = None
        for candidate in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        x = 0
        for col, lbl in columns:
            cw = col.shape[1]
            bbox = draw.textbbox((0, 0), lbl, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x + (cw - tw) // 2, (label_h - th) // 2 - 2),
                      lbl, fill=(0, 0, 0), font=font)
            x += cw
        panel = np.concatenate([panel, np.asarray(strip)], axis=0)

        if args.side_by_side is not None:
            Image.fromarray(panel).save(args.side_by_side)
            print(f"Wrote side-by-side panel to {args.side_by_side}")

        if args.show:
            import matplotlib.pyplot as plt
            n = len(columns)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
            for ax, (col, lbl) in zip(axes, columns):
                ax.imshow(col); ax.set_title(lbl); ax.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
