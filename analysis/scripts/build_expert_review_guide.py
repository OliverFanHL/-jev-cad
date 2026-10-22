"""Build a print-friendly CAD expert-review interface as a static PDF figure.

The design intentionally looks like an application UI while remaining fully
non-interactive for manuscript use.  The page background is white so the
figure blends naturally into a white paper page.
"""

from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/expert_review_interface.pdf"
RENDERS = ROOT.parent / "apps/expert_review/public/renders"
W, H = 518.4, 480.0  # 7.2 x 6.67 inches; compact height for manuscript use


def register_fonts():
    regular_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/mnt/data/C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/mnt/data/C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
    ]
    regular = next((p for p in regular_candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    if not regular or not bold:
        raise FileNotFoundError("Arial (or Arimo fallback) could not be found.")
    pdfmetrics.registerFont(TTFont("Arial", str(regular)))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold)))


register_fonts()

# Restrained typography: four sizes only.
FONT_XS = 7.2
FONT_SM = 8.0
FONT_LABEL = 8.7
FONT_TITLE = 9.6

# Print-friendly palette.  White is the global background; color is localized.
navy = HexColor("#17324D")
blue = HexColor("#14759A")
blue_dark = HexColor("#0F5F81")
teal = HexColor("#278A8E")
slate = HexColor("#61798A")
muted = HexColor("#8495A2")
line = HexColor("#BFD0DB")
line_soft = HexColor("#DDE7ED")
line_faint = HexColor("#EDF2F5")
white = HexColor("#FFFFFF")
soft = HexColor("#FAFCFD")
pale_blue = HexColor("#EAF5F9")
pale_teal = HexColor("#EAF6F5")
pale_gray = HexColor("#F1F5F7")
green = HexColor("#2F9278")
shadow = HexColor("#E8EEF2")

MARGIN = 20.0
CONTENT_W = W - 2 * MARGIN
PAIR_GAP = 34.0
PAIR_W = (CONTENT_W - PAIR_GAP) / 2


def label(c, x, y, text, size=FONT_SM, bold=False, color=navy):
    c.setFont("Arial-Bold" if bold else "Arial", size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def center(c, x, y, text, size=FONT_SM, bold=False, color=navy):
    c.setFont("Arial-Bold" if bold else "Arial", size)
    c.setFillColor(color)
    c.drawCentredString(x, y, text)


def right(c, x, y, text, size=FONT_SM, bold=False, color=navy):
    c.setFont("Arial-Bold" if bold else "Arial", size)
    c.setFillColor(color)
    c.drawRightString(x, y, text)


def rounded(c, x, y, w, h, fill=white, stroke=line, radius=7,
            line_width=0.75, shadowed=False):
    if shadowed:
        c.setFillColor(shadow)
        c.roundRect(x + 1.0, y - 1.0, w, h, radius, stroke=0, fill=1)
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def chip(c, x, y, w, h, text, fill, color=blue_dark, stroke=None,
         size=FONT_XS, bold=True):
    c.setFillColor(fill)
    c.setStrokeColor(stroke if stroke is not None else fill)
    c.setLineWidth(0.65)
    c.roundRect(x, y, w, h, h / 2, stroke=1 if stroke is not None else 0, fill=1)
    center(c, x + w / 2, y + (h - size) / 2 + 0.9, text, size, bold, color)


def check(c, cx, cy, color=green, scale=1.0):
    c.setStrokeColor(color)
    c.setLineWidth(1.2 * scale)
    c.setLineCap(1)
    c.line(cx - 2.5 * scale, cy, cx - 0.5 * scale, cy - 2.0 * scale)
    c.line(cx - 0.5 * scale, cy - 2.0 * scale, cx + 3.1 * scale, cy + 2.2 * scale)


def question_icon(c, cx, cy):
    c.setFillColor(blue_dark)
    c.circle(cx, cy, 9.5, stroke=0, fill=1)
    center(c, cx, cy - 2.8, "?", 8.8, True, white)


def magnifier(c, cx, cy, color=slate):
    c.setStrokeColor(color)
    c.setLineWidth(1.0)
    c.circle(cx - 1.2, cy + 1.0, 3.3, stroke=1, fill=0)
    c.line(cx + 1.4, cy - 1.5, cx + 4.0, cy - 4.0)


def expand_icon(c, cx, cy, color=slate):
    c.setStrokeColor(color)
    c.setLineWidth(0.9)
    s = 3.4
    c.line(cx - s, cy + s, cx - 0.8, cy + s)
    c.line(cx - s, cy + s, cx - s, cy + 0.8)
    c.line(cx + s, cy + s, cx + 0.8, cy + s)
    c.line(cx + s, cy + s, cx + s, cy + 0.8)
    c.line(cx - s, cy - s, cx - 0.8, cy - s)
    c.line(cx - s, cy - s, cx - s, cy - 0.8)
    c.line(cx + s, cy - s, cx + 0.8, cy - s)
    c.line(cx + s, cy - s, cx + s, cy - 0.8)


def rotate_icon(c, cx, cy, color=slate):
    c.setStrokeColor(color)
    c.setLineWidth(0.95)
    c.arc(cx - 4.2, cy - 4.2, cx + 4.2, cy + 4.2, 35, 275)
    c.line(cx + 3.6, cy + 2.6, cx + 3.8, cy - 0.3)
    c.line(cx + 3.6, cy + 2.6, cx + 0.9, cy + 1.7)


def pencil_icon(c, cx, cy, color=slate):
    c.setStrokeColor(color)
    c.setLineWidth(1.0)
    c.line(cx - 3.2, cy - 2.4, cx + 2.7, cy + 3.5)
    c.line(cx - 3.9, cy - 3.2, cx - 2.4, cy - 2.8)
    c.line(cx + 2.1, cy + 4.1, cx + 3.3, cy + 2.9)


def arrow_compare(c, cx, cy):
    c.setFillColor(white)
    c.setStrokeColor(line)
    c.setLineWidth(0.85)
    c.circle(cx, cy, 10.5, stroke=1, fill=1)
    c.setStrokeColor(blue_dark)
    c.setLineWidth(1.15)
    c.setLineCap(1)
    c.line(cx - 5.0, cy + 2.0, cx + 3.8, cy + 2.0)
    c.line(cx + 3.8, cy + 2.0, cx + 1.1, cy + 4.0)
    c.line(cx + 3.8, cy + 2.0, cx + 1.1, cy + 0.0)
    c.line(cx + 5.0, cy - 2.0, cx - 3.8, cy - 2.0)
    c.line(cx - 3.8, cy - 2.0, cx - 1.1, cy - 4.0)
    c.line(cx - 3.8, cy - 2.0, cx - 1.1, cy + 0.0)


def chevron(c, cx, cy, direction="right", color=blue_dark, scale=1.0):
    dx, dy = 2.8 * scale, 4.1 * scale
    c.setStrokeColor(color)
    c.setLineWidth(1.35 * scale)
    c.setLineCap(1)
    if direction == "right":
        c.line(cx - dx, cy - dy, cx + dx, cy)
        c.line(cx + dx, cy, cx - dx, cy + dy)
    else:
        c.line(cx + dx, cy - dy, cx - dx, cy)
        c.line(cx - dx, cy, cx + dx, cy + dy)


def nav_control(c, x, y, text, direction, primary=False):
    r = 11.0
    cx, cy = x + r, y + r
    if primary:
        c.setFillColor(blue_dark)
        c.setStrokeColor(blue_dark)
        c.circle(cx, cy, r, stroke=1, fill=1)
        chevron(c, cx, cy, direction, white, 1.0)
    else:
        c.setFillColor(white)
        c.setStrokeColor(line)
        c.setLineWidth(0.9)
        c.circle(cx, cy, r, stroke=1, fill=1)
        chevron(c, cx, cy, direction, blue_dark, 0.95)
    label(c, x + 29.0, y + 7.7, text, FONT_SM, True, blue_dark)


def image_toolbar(c, x, y):
    for i, icon in enumerate((magnifier, rotate_icon, expand_icon)):
        bx = x + i * 20.0
        c.setFillColor(white)
        c.setStrokeColor(line_soft)
        c.setLineWidth(0.65)
        c.roundRect(bx, y, 16.0, 16.0, 4.0, stroke=1, fill=1)
        icon(c, bx + 8.0, y + 8.0, slate)


def viewer_grid(c, x, y, w, h):
    # Very faint CAD-view grid; localized so the manuscript page remains white.
    c.saveState()
    c.setStrokeColor(line_faint)
    c.setLineWidth(0.35)
    step = 16.0
    xx = x + step
    while xx < x + w:
        c.line(xx, y + 5, xx, y + h - 5)
        xx += step
    yy = y + step
    while yy < y + h:
        c.line(x + 5, yy, x + w - 5, yy)
        yy += step
    c.restoreState()


def part_viewer(c, x, title, image_name, role="query", query_id=None):
    y, h = 228.0, 124.0
    accent = blue if role == "query" else teal
    tint = pale_blue if role == "query" else pale_teal

    rounded(c, x, y, PAIR_W, h, white, line, 7, 0.8, True)

    # Compact viewer header.
    c.setFillColor(soft)
    c.roundRect(x + 0.5, y + h - 25.5, PAIR_W - 1, 25.0, 6.5, stroke=0, fill=1)
    c.rect(x + 0.5, y + h - 25.5, PAIR_W - 1, 6.0, stroke=0, fill=1)
    c.setFillColor(accent)
    c.roundRect(x + 10.0, y + h - 18.0, 4.0, 8.5, 2.0, stroke=0, fill=1)
    label(c, x + 21.0, y + h - 18.0, title, FONT_LABEL, True)

    if query_id:
        chip(c, x + PAIR_W - 54.0, y + h - 20.0, 42.0, 14.0,
             query_id, tint, accent, size=FONT_XS)

    # CAD viewport.
    well_x, well_y = x + 10.0, y + 25.0
    well_w, well_h = PAIR_W - 20.0, h - 59.0
    c.setFillColor(soft)
    c.roundRect(well_x, well_y, well_w, well_h, 5.0, stroke=0, fill=1)
    viewer_grid(c, well_x, well_y, well_w, well_h)

    img_path = RENDERS / image_name
    if not img_path.exists():
        img_path = FALLBACK_RENDERS / ("img_0.png" if role == "query" else "img_1.png")
    image = Image.open(img_path).convert("RGB")
    iw, ih = image.size
    max_w, max_h = 76.0, 50.0
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(
        ImageReader(image),
        well_x + (well_w - dw) / 2,
        well_y + (well_h - dh) / 2,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
        mask="auto",
    )

    # Static viewer tools.
    chip(c, x + 10.0, y + 5.5, 52.0, 15.0, "STEP view", pale_gray, slate,
         size=FONT_XS, bold=False)
    image_toolbar(c, x + PAIR_W - 68.0, y + 5.0)


def score_selector(c):
    title_y = 199.0
    label(c, MARGIN, title_y, "Engineering relevance", FONT_LABEL, True)
    chip(c, W - MARGIN - 73.0, title_y - 4.0, 73.0, 17.0,
         "choose 0-3", pale_blue, blue_dark, size=FONT_XS, bold=False)

    x0 = MARGIN + 35.0
    x1 = W - MARGIN - 35.0
    cy = 162.0
    xs = [x0 + i * (x1 - x0) / 3 for i in range(4)]

    # Static stepper rail: visually interactive, but intentionally non-fillable.
    c.setStrokeColor(line_soft)
    c.setLineWidth(3.0)
    c.setLineCap(1)
    c.line(xs[0], cy, xs[-1], cy)

    scores = [
        ("0", "Unrelated", "form and role"),
        ("1", "Broad form", "or role overlap"),
        ("2", "Similar shape", "or fit"),
        ("3", "Close", "substitute"),
    ]

    for i, (num, top, bottom) in enumerate(scores):
        # Outer halo and inner selectable node.
        c.setFillColor(white)
        c.setStrokeColor(line)
        c.setLineWidth(0.9)
        c.circle(xs[i], cy, 12.0, stroke=1, fill=1)
        c.setFillColor(pale_blue)
        c.circle(xs[i], cy, 8.0, stroke=0, fill=1)
        center(c, xs[i], cy - 2.7, num, FONT_SM, True, blue_dark)

        center(c, xs[i], cy - 27.0, top, FONT_SM, True, navy)
        center(c, xs[i], cy - 38.0, bottom, FONT_XS, False, slate)


def notes_field(c):
    label(c, MARGIN, 108.0, "Notes", FONT_LABEL, True)
    label(c, MARGIN + 31.0, 108.0, "optional", FONT_XS, False, muted)
    right(c, W - MARGIN, 108.0, "Add a concise engineering observation", FONT_XS, False, muted)

    y, h = 70.0, 32.0
    rounded(c, MARGIN, y, CONTENT_W, h, white, line, 6, 0.8, False)
    c.setFillColor(pale_blue)
    c.circle(MARGIN + 17.0, y + h / 2, 8.0, stroke=0, fill=1)
    pencil_icon(c, MARGIN + 17.0, y + h / 2, blue_dark)
    c.setStrokeColor(line_soft)
    c.setLineWidth(0.6)
    c.line(MARGIN + 34.0, y + 7.0, MARGIN + 34.0, y + h - 7.0)
    label(c, MARGIN + 43.0, y + 12.0,
          "For example: the bores align, but the outer shape differs.", FONT_SM, False, muted)


def build_pdf():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Blinded CAD expert review - print figure")

    # No colored full-page background: the PDF blends into a white manuscript page.
    c.setFillColor(white)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Header.
    c.setFillColor(blue)
    c.circle(MARGIN + 3.5, 461.5, 3.2, stroke=0, fill=1)
    label(c, MARGIN + 12.0, 458.0, "INDEPENDENT EXPERT A", FONT_SM, True, slate)
    label(c, MARGIN, 437.0, "Pair 1 of 6,645", FONT_TITLE, True)

    chip(c, W - MARGIN - 132.0, 450.0, 65.0, 18.0, "All saved", white, green,
         stroke=line, size=FONT_XS, bold=False)
    check(c, W - MARGIN - 120.0, 459.0, green, 0.7)
    chip(c, W - MARGIN - 60.0, 450.0, 60.0, 18.0, "Leave", white, slate,
         stroke=line, size=FONT_XS, bold=False)

    # Progress indicator.
    c.setStrokeColor(line_soft)
    c.setLineWidth(3.0)
    c.setLineCap(1)
    c.line(MARGIN, 424.0, W - MARGIN, 424.0)
    label(c, MARGIN, 414.0, "0 of 6,645 saved", FONT_XS, True, blue_dark)
    right(c, W - MARGIN, 414.0, "Review progress", FONT_XS, False, muted)

    # Task card - localized background only.
    qy, qh = 363.0, 42.0
    rounded(c, MARGIN, qy, CONTENT_W, qh, white, line, 7, 0.75, False)
    c.setFillColor(pale_blue)
    c.roundRect(MARGIN, qy, 5.0, qh, 2.5, stroke=0, fill=1)
    question_icon(c, MARGIN + 17.0, qy + qh / 2 + 1.0)
    label(c, MARGIN + 36.0, qy + 26.0,
          "Could the candidate serve the same purpose as the query part?", FONT_LABEL, True)
    chip(c, MARGIN + 36.0, qy + 6.0, 43.0, 14.0, "Shape", pale_blue, blue_dark, size=FONT_XS)
    chip(c, MARGIN + 84.0, qy + 6.0, 32.0, 14.0, "Fit", pale_blue, blue_dark, size=FONT_XS)
    chip(c, MARGIN + 121.0, qy + 6.0, 76.0, 14.0, "Mechanical role", pale_blue, blue_dark, size=FONT_XS)

    # Pair viewers.
    left_x = MARGIN
    right_x = MARGIN + PAIR_W + PAIR_GAP
    part_viewer(c, left_x, "Query part", "4334353858dd5d87a859.webp", "query", "query 1")
    part_viewer(c, right_x, "Candidate part", "a674bbc7af990dd6ac77.webp", "candidate")
    arrow_compare(c, MARGIN + PAIR_W + PAIR_GAP / 2, 290.0)

    # Static interaction-like controls.
    score_selector(c)
    notes_field(c)

    # Navigation.
    c.setStrokeColor(line_soft)
    c.setLineWidth(0.7)
    c.line(MARGIN, 58.0, W - MARGIN, 58.0)
    nav_control(c, MARGIN, 23.0, "Previous", "left", False)
    nav_control(c, MARGIN + 106.0, 23.0, "Next unrated", "right", True)
    right(c, W - MARGIN, 31.0, "Select a score before continuing", FONT_XS, False, muted)

    c.showPage()
    c.save()


PREVIEW = Path("/mnt/data/expert_review_interface_v10_compact_preview.png")


def build_preview(dpi=600):
    import fitz
    doc = fitz.open(str(OUT))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    pix.save(str(PREVIEW))
    doc.close()


def main():
    build_pdf()
    # build_preview()
    print(OUT)
    print(PREVIEW)


if __name__ == "__main__":
    main()
