"""Generate the PolViz icon (a Poincare sphere) as .ico and .png.

Usage:  python tools/make_icon.py        (needs numpy and Pillow)
Writes polviz/assets/polviz.ico (16-256 px, for Windows / PyInstaller) and
polviz/assets/polviz.png (512 px, used as the window icon).

Sizes up to 48 px use a simplified drawing so they stay readable.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "polviz" / "assets"
SS = 2048                     # supersampled canvas size
R = 0.86                      # sphere radius as a fraction of half the canvas
AZ, EL = np.radians(55), np.radians(20)   # viewing direction

# Camera frame: right, up, towards-viewer
V = np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)])
RIGHT = np.array([-np.sin(AZ), np.cos(AZ), 0.0])
UP = np.cross(V, RIGHT)

RED, GREEN, BLUE = (240, 90, 90), (90, 210, 110), (110, 160, 255)
ORANGE, WHITE = (255, 160, 60), (255, 255, 255)


def to_px(p):
    """3D point on/near the unit sphere -> canvas pixel coordinates."""
    half = SS / 2
    return (half + R * half * np.dot(p, RIGHT), half - R * half * np.dot(p, UP))


def draw_sphere(img, gloss=0.35):
    n = SS
    y, x = np.mgrid[0:n, 0:n]
    sx = (x + 0.5 - n / 2) / (R * n / 2)
    sy = -(y + 0.5 - n / 2) / (R * n / 2)
    r2 = sx ** 2 + sy ** 2
    inside = r2 <= 1
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    diffuse = np.clip(sx * light[0] + sy * light[1] + nz * light[2], 0, 1)
    refl_z = 2 * diffuse * nz - light[2]          # specular term (viewer along +z)
    spec = np.clip(refl_z, 0, 1) ** 60
    dark, bright = np.array([14, 30, 80]), np.array([95, 160, 245])
    shade = 0.15 + 0.85 * diffuse
    rgb = dark + (bright - dark) * shade[..., None] + 255 * gloss * spec[..., None]
    rim = np.clip((r2 - 0.80) / 0.20, 0, 1) ** 2   # soft bright rim, helps on dark taskbars
    rgb = rgb + np.array([60, 90, 140]) * rim[..., None] * 0.5
    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255)
    rgba[..., 3] = np.where(inside, 255, 0)
    img.alpha_composite(Image.fromarray(rgba, "RGBA"))


def polyline(draw, pts3d, color, width, back_alpha=70, front_alpha=255):
    """Draw a 3D curve, dimming the parts behind the sphere."""
    pts3d = np.asarray(pts3d)
    depth = pts3d @ V
    for a, b, d in zip(pts3d[:-1], pts3d[1:], depth[:-1]):
        alpha = front_alpha if d >= 0 else back_alpha
        draw.line([to_px(a), to_px(b)], fill=color + (alpha,), width=width)


def circle(normal_axis, t):
    """Great circle perpendicular to one of the coordinate axes."""
    c, s = np.cos(t), np.sin(t)
    z = np.zeros_like(t)
    return {"z": np.column_stack([c, s, z]),
            "y": np.column_stack([c, z, s]),
            "x": np.column_stack([z, c, s])}[normal_axis]


def dot(draw, p, radius, fill, ring=None, ring_w=0):
    x, y = to_px(p)
    if ring:
        rr = radius + ring_w
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=ring + (255,))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill + (255,))


def render(detailed: bool) -> Image.Image:
    img = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    draw_sphere(img, 0.35 if detailed else 0.15)
    overlay = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    t = np.linspace(0, 2 * np.pi, 721)

    if detailed:
        for lat in (-50, 50):            # two extra circles of latitude
            z, rr = np.sin(np.radians(lat)), np.cos(np.radians(lat))
            pts = np.column_stack([rr * np.cos(t), rr * np.sin(t), np.full_like(t, z)])
            polyline(d, pts, (210, 225, 255), 10, 35, 110)
        polyline(d, circle("x", t), (210, 225, 255), 10, 35, 110)
        polyline(d, circle("y", t), (225, 235, 255), 12, 45, 150)
        polyline(d, circle("z", t), (235, 242, 255), 16, 60, 230)   # equator
    else:                                 # bold strokes so they survive 16-48 px
        polyline(d, circle("z", t), (235, 242, 255), 90, 0, 235)

    if detailed:                          # Stokes axes, same colours as the app
        s = np.linspace(-1.12, 1.12, 200)
        for axis, color in ((0, RED), (1, GREEN), (2, BLUE)):
            pts = np.zeros((len(s), 3))
            pts[:, axis] = s
            polyline(d, pts, color, 20, 140, 255)

    # A waveplate-like trajectory: from H up towards the north pole
    tt = np.linspace(0, np.radians(68), 200)
    arc = np.column_stack([np.cos(tt), np.zeros_like(tt), np.sin(tt)])
    if detailed:
        polyline(d, arc, ORANGE, 30)
        dot(d, arc[0], 34, WHITE)
        dot(d, arc[-1], 50, WHITE, ORANGE, 22)
    else:
        polyline(d, arc, ORANGE, 150)
        dot(d, arc[-1], 150, WHITE, ORANGE, 90)

    img.alpha_composite(overlay)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    big, small = render(True), render(False)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [(small if s <= 48 else big).resize((s, s), Image.LANCZOS) for s in sizes]
    frames[-1].save(OUT / "polviz.ico", sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    big.resize((512, 512), Image.LANCZOS).save(OUT / "polviz.png")
    print(f"Wrote {OUT / 'polviz.ico'} and {OUT / 'polviz.png'}")


if __name__ == "__main__":
    main()
