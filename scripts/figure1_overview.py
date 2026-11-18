"""Figure 1: GateLink architecture (vector PDF + PNG preview), 178 mm wide.
Color is used only as an accent: the two parts GateLink adds (signal extractor and gate) are blue, the color of
GateLink in Figure 2; everything else is neutral. Every path is named by a text label and the two kinds of
secondary path differ by line style (dashed = fallback, dotted = offline), so nothing depends on color.
Usage: python scripts/figure1_overview.py <out stem>   (e.g. figures/figure1)"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = sys.argv[1]
INK, INK2, EDGE, FILL = "#0b0b0b", "#52514e", "#9a9893", "#f7f6f3"
BLUE, BLUE_FILL = "#2a78d6", "#e7f1fb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})

W, H = 178 / 25.4, 106 / 25.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 178)
ax.set_ylim(0, 106)
ax.axis("off")


def box(x, y, w, h, title, sub=None, accent=False, dotted=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                       lw=1.2 if accent else 0.8, ec=BLUE if accent else EDGE,
                       fc=BLUE_FILL if accent else FILL, ls=(0, (1.2, 1.4)) if dotted else "-")
    ax.add_patch(p)
    cy = y + h / 2
    if sub:
        n = sub.count("\n")
        ax.text(x + w / 2, cy + 2.3 + 1.3 * n, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, cy - 2.6 + 0.4 * n, sub, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=cy)


def document(x, y, w, h, title, sub):
    """Standard flowchart 'document' symbol: straight top and sides, wavy bottom edge."""
    t = np.linspace(0, 1, 60)
    bottom = [(x + w * u, y + 1.6 + 1.6 * np.sin(2 * np.pi * u)) for u in t]   # one full wave
    pts = [(x, y + h), (x + w, y + h)] + bottom[::-1]
    ax.add_patch(Polygon(pts, closed=True, fc=FILL, ec=EDGE, lw=0.8, joinstyle="round"))
    cy = y + h / 2 + 0.8
    ax.text(x + w / 2, cy + 2.3, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, cy - 2.6, sub, ha="center", va="center", fontsize=6.3, color=INK2)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2 + 0.8)


def data(x, y, w, h, title, sub, slant=3.0):
    """Standard flowchart 'data' (input/output) symbol: a parallelogram."""
    pts = [(x + slant, y + h), (x + w, y + h), (x + w - slant, y), (x, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=FILL, ec=EDGE, lw=0.8, joinstyle="round"))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 2.3, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, cy - 2.6, sub, ha="center", va="center", fontsize=6.3, color=INK2)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    return dict(l=x + slant / 2, r=x + w - slant / 2, b=y, t=y + h, cx=x + w / 2, cy=cy)


def database(x, y, w, h, title, sub, e=3.2):
    """Standard flowchart 'stored data' symbol: a cylinder."""
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=FILL, ec="none"))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=FILL, ec="none"))
    ax.add_patch(Arc((x + w / 2, y + e / 2), w, e, theta1=180, theta2=360, ec=EDGE, lw=0.8))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=EDGE, lw=0.8)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=EDGE, lw=0.8)
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=FILL, ec=EDGE, lw=0.8))
    cy = y + h / 2 - 0.8
    ax.text(x + w / 2, cy + 2.2, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, cy - 2.4, sub, ha="center", va="center", fontsize=6.3, color=INK2)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=cy)


def arrow(p, q, ls="-", color=INK2, lw=0.9, label=None, lpos=0.5, loff=(0, 1.8), ha="center", rad=0.0):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha=ha, va="center", fontsize=6.5, color=INK, style="italic")


def poly(points, ls="-", color=INK2, lw=0.9, head=True):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(points[-2], points[-1], ls=ls, color=color, lw=lw) if head else ax.plot(xs[-2:], ys[-2:], color=color, lw=lw, ls=ls)


# ---- top row: input -> candidates -> signals
doc = document(3, 70, 26, 16, "Document d", "abstract or full text")
men = data(31.5, 70, 31, 16, "Mention m", "span and entity type")
cgen = box(65, 70, 54, 16, "Candidate generator C", "exact match + zero-shot SapBERT\n\u2192 top-20 concepts with scores")
phi = box(124, 70, 51, 16, "Signal extractor \u03c6", "features of Table 1\nfrom m, d, C and V", accent=True)
arrow((doc["r"], men["cy"]), (men["l"], men["cy"]))
arrow((men["r"], men["cy"]), (cgen["l"], cgen["cy"]))
arrow((cgen["r"], cgen["cy"]), (phi["l"], phi["cy"]))

# ---- vocabulary (stored data), feeding the candidate generator and the signal extractor
voc = database(90, 91, 62, 13, "Vocabulary V", "names, synonyms and hierarchy")
arrow((cgen["cx"] + 8, voc["b"] + 0.1), (cgen["cx"] + 8, cgen["t"]))
arrow((phi["cx"] - 8, voc["b"] + 0.1), (phi["cx"] - 8, phi["t"]))

# ---- gate (diamond), below the signal extractor
gx, gy, gw, gh = phi["cx"], 49, 25, 11
ax.add_patch(Polygon([(gx - gw, gy), (gx, gy + gh), (gx + gw, gy), (gx, gy - gh)], closed=True,
                     fc=BLUE_FILL, ec=BLUE, lw=1.2))
ax.text(gx, gy + 2.4, "Gate g", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
ax.text(gx, gy - 2.3, "estimates P(top-1 wrong)", ha="center", va="center", fontsize=6.3, color=INK2)
arrow((phi["cx"], phi["b"]), (gx, gy + gh))

# ---- the two decisions
llm = box(65, 42, 46, 14, "LLM reranker L", "zero-shot Gemma 4\n\u2192 candidate number or NIL")
keep = box(124, 17, 51, 12, "Keep the retriever's top-1", "answer = top candidate of C")
arrow((gx - gw, gy), (llm["r"], gy), lw=1.3, color=INK, label="g \u2265 \u03c4", lpos=0.5, loff=(0, 2.4))
arrow((gx, gy - gh), (gx, keep["t"]), lw=1.3, color=INK, label="g < \u03c4", lpos=0.5, loff=(1.6, 0), ha="left")

# invalid answer -> fallback (dashed)
fx = llm["r"] - 6
poly([(fx, llm["b"]), (fx, keep["cy"]), (keep["l"], keep["cy"])], ls=(0, (3, 2)))
ax.text(fx + 1.5, 33.5, "invalid answer", ha="left", va="center", fontsize=6.5, color=INK, style="italic")

# ---- output
out = data(30, 3, 88, 10, "Concept identifier or NIL", None)
poly([(llm["cx"] - 6, llm["b"]), (llm["cx"] - 6, out["t"])])
poly([(keep["cx"], keep["b"]), (keep["cx"], out["cy"]), (out["r"], out["cy"])])

# ---- offline step (dotted), left of the LLM
off = box(3, 26, 59, 38, "Offline: train g and choose \u03c4", None, dotted=True)
ax.texts[-1].set_position((off["cx"], off["t"] - 4))
B = "bold"
algo = [  # (line number, indent, [(text, weight)])
    ("", 0, [("Input: ", B), ("development mentions D; rule A, B or R", None)]),
    ("", 0, [("Output: ", B), ("gate g and threshold \u03c4", None)]),
    ("1", 0, [("for each ", B), ("m in D ", None), ("do", B)]),
    ("2", 1, [("z(m) \u2190 1 if the retriever's top-1 is wrong", None)]),
    ("3", 0, [("train calibrated g on (\u03c6(m), z(m))", None)]),
    ("4", 0, [("\u011d(m) \u2190 out-of-fold score (document folds)", None)]),
    ("5", 0, [("if ", B), ("rule is A or R ", None), ("then", B)]),
    ("6", 1, [("run L once on every m in D", None)]),
    ("7", 0, [("\u03c4 \u2190 threshold on \u011d by rule A, B or R", None)]),
    ("8", 0, [("return ", B), ("g, \u03c4", None)]),
]
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
y0, dy = off["t"] - 9.5, 2.85
for i, (num, ind, parts) in enumerate(algo):
    y = y0 - i * dy
    if num:
        ax.text(6, y, num, ha="left", va="center", fontsize=6.4, color=INK2)
    x = (6 if i < 2 else 9.5) + 4.2 * ind
    for txt, w in parts:
        t = ax.text(x, y, txt, ha="left", va="center", fontsize=6.4, color=INK,
                    fontweight=w or "normal")
        bb = t.get_window_extent(renderer=rend)
        x = inv.transform((bb.x1, bb.y0))[0]
    if i == 1:
        ax.plot([5.5, off["r"] - 2.5], [y - 1.45, y - 1.45], color=EDGE, lw=0.5)
ex = gx - 15
poly([(off["r"], 60), (ex, 60), (ex, gy + gh * (1 - 15 / gw))], ls=(0, (1.2, 1.4)))
ax.text(off["r"] + 2, 62, "trained gate and \u03c4", ha="left", va="center", fontsize=6.5, color=INK, style="italic")

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=300)
print("wrote", OUT)
