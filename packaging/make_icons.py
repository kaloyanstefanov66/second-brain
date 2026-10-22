"""Generate the app icon (PNG, Windows .ico, macOS .icns) from code. Run: python packaging/make_icons.py"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "icons"
S = 4096  # draw large, downsample for smooth edges


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def icon() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # macOS-style squircle-ish rounded square with a margin, vertical violet gradient
    pad, radius = int(S * 0.09), int(S * 0.2)
    grad = Image.new("RGBA", (S, S))
    top, bottom = (124, 108, 240), (74, 58, 167)
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        gd.line([(0, y), (S, y)], fill=lerp(top, bottom, y / S) + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], radius, fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    u = (S - 2 * pad) / 32  # same 32-unit grid as the favicon
    p = lambda x, y: (pad + x * u, pad + y * u)
    nodes = [(10, 11, 3.2), (22, 10, 3.2), (16, 22.5, 3.6)]
    white, w = (255, 255, 255, 255), int(2.6 * u)
    for (x1, y1, _), (x2, y2, _) in [(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[2])]:
        d.line([p(x1, y1), p(x2, y2)], fill=white, width=w)
    for x, y, r in nodes:
        cx, cy = p(x, y)
        d.ellipse([cx - r * u, cy - r * u, cx + r * u, cy + r * u], fill=white)
        ri = (r - 1.5) * u
        d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=lerp(top, bottom, cy / S) + (255,))
    return img.resize((1024, 1024), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    big = icon()
    big.save(OUT / "icon.png")
    big.resize((256, 256), Image.LANCZOS).save(OUT / "icon-256.png")
    big.save(OUT / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(OUT / "icon.icns")
    print("wrote", ", ".join(sorted(f.name for f in OUT.iterdir())))
