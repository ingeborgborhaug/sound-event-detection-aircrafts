import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import textwrap


def fit_text(ax, fig, text_str, x, y, box_width, box_height, initial_fs, ha="center", va="center", family="serif", pad_px=6):
    """
    Place text at (x,y) inside a box of size (box_width, box_height) in data units.
    If the single-line text doesn't fit horizontally, wrap into multiple lines to fit.
    This function preserves the `initial_fs` fontsize (does not shrink) unless no
    wrapping fits; then it will minimally shrink as a fallback.
    """
    fs = int(initial_fs)
    # helper to measure a given text string (may contain '\n') at fontsize fs
    def measure(txt_str, fontsize):
        txt_obj = ax.text(x, y, txt_str, ha=ha, va=va, fontsize=fontsize, family=family)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = txt_obj.get_window_extent(renderer=renderer)
        txt_obj.remove()
        return bbox.width, bbox.height

    # compute box pixel dimensions
    left_disp = ax.transData.transform((x - box_width / 2.0, y))[0]
    right_disp = ax.transData.transform((x + box_width / 2.0, y))[0]
    box_pix_w = abs(right_disp - left_disp)
    top_disp = ax.transData.transform((x, y + box_height / 2.0))[1]
    bottom_disp = ax.transData.transform((x, y - box_height / 2.0))[1]
    box_pix_h = abs(top_disp - bottom_disp)

    # approximate line height in pixels for fontsize
    line_height_px = fs * fig.dpi / 72.0 * 1.2

    max_lines = max(1, int((box_pix_h - 2 * pad_px) // line_height_px))

    # Try wrapping into 1..max_lines lines
    best_txt = None
    for n_lines in range(1, max_lines + 1):
        # initial guess for chars per line
        target = max(1, int(len(text_str) / n_lines))
        # try increasing width until wrapped into <= n_lines lines
        for w in range(target, len(text_str) + 1):
            wrapped = textwrap.fill(text_str, width=w, break_long_words=False, break_on_hyphens=False)
            lines = wrapped.count('\n') + 1
            if lines <= n_lines:
                # measure
                bw, bh = measure(wrapped, fs)
                if bw <= (box_pix_w - pad_px) and bh <= (box_pix_h - pad_px):
                    best_txt = ax.text(x, y, wrapped, ha=ha, va=va, fontsize=fs, family=family)
                    return best_txt
                # if width fits but height doesn't, no need to try larger w for this n_lines
                if bw <= (box_pix_w - pad_px) and bh > (box_pix_h - pad_px):
                    break

    # Fallback: try reducing fontsize until it fits on a single line (rare)
    txt = ax.text(x, y, text_str, ha=ha, va=va, fontsize=fs, family=family)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = txt.get_window_extent(renderer=renderer)
    while (tb.width > (box_pix_w - pad_px) or tb.height > (box_pix_h - pad_px)) and fs > 4:
        fs -= 1
        txt.set_fontsize(fs)
        fig.canvas.draw()
        tb = txt.get_window_extent(renderer=renderer)

    return txt


def draw_diagram(fontsize=16, out_file="yamnet_diagram.png"):
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Outer model box
    ax.add_patch(Rectangle((2.6, 0.7), 8.8, 5.0, fill=False, lw=4, ec="#2f3a45", joinstyle="round"))

    # Input box
    input_box = Rectangle((0.1, 2.0), 2.0, 1.7, fc="#eeeeee", ec="black", lw=1.5, joinstyle="round")
    ax.add_patch(input_box)
    fit_text(ax, fig, "Audio input (16kHz)", 0.1 + 2.0 / 2.0, 2.0 + 1.7 / 2.0, 2.0, 1.7, fontsize, family="Times New Roman")

    # Feature extraction text and box
    ax.text(4.5, 5.15, "Feature extraction", ha="center", fontsize=fontsize, family="Times New Roman")
    feat_box = Rectangle((3.5, 2.0), 2.0, 1.7, fc="#eeeeee", ec="black", lw=1.5, joinstyle="round")
    ax.add_patch(feat_box)
    fit_text(ax, fig, "Log-mel spectrogram", 3.5 + 2.0 / 2.0, 2.0 + 1.7 / 2.0, 2.0, 1.7, fontsize, family="Times New Roman")

    # MobileNet title
    ax.text(7.7, 5.15, "Mobile_v1 model", ha="center", fontsize=fontsize, family="Times New Roman")

    # Output title
    ax.text(10.25, 5.15, "Output layer", ha="center", fontsize=fontsize, family="Times New Roman")

    # Arrows between major blocks (input -> feature -> Mobile -> output)
    for x1, y1, x2, y2 in [
        (2.1, 2.85, 3.1, 2.85),
        (5.6, 2.85, 6.5, 2.85),
        (9.0, 2.85, 9.8, 2.85),
        (10.9, 2.85, 11.7, 2.85),
    ]:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="Simple,head_width=0.45,head_length=0.28,tail_width=0.18",
            fc="black", ec="black", lw=1.0, mutation_scale=10
        ))

    # Neural network nodes
    x_cols = [7.0, 7.8, 8.6]
    y_nodes = [4.45, 3.75, 2.25, 1.55]

    for x in x_cols:
        for y in y_nodes:
            ax.add_patch(Circle((x, y), 0.23, fc="#1769aa", ec="black", lw=1.5))

    # Connections (arrows between nodes)
    for i in range(len(x_cols)-1):
        for y1 in y_nodes:
            for y2 in y_nodes:
                if abs(y1 - y2) <= 2.2:
                    start = (x_cols[i] + 0.23, y1)
                    end = (x_cols[i+1] - 0.23, y2)
                    ax.add_patch(FancyArrowPatch(
                        start, end,
                        arrowstyle="Simple,head_width=0.28,head_length=0.22,tail_width=0.08",
                        mutation_scale=6, lw=0.6, fc="black", ec="black"
                    ))

    # Ellipsis dots in network
    for x in x_cols:
        for y in [3.15, 2.95, 2.75]:
            ax.add_patch(Circle((x, y), 0.055, fc="black", ec="black"))

    # Output layer nodes
    for y in [4.45, 3.75, 2.25, 1.55]:
        ax.add_patch(Circle((10.2, y), 0.23, fc="#cfe1f9", ec="black", lw=1.5))

    for y in [3.15, 2.95, 2.75]:
        ax.add_patch(Circle((10.2, y), 0.055, fc="black", ec="black"))

    # Output text (placed outside the small output column, no fitting needed)
    ax.text(11.95, 2.85, "512 Class Scores", ha="left", va="center", fontsize=fontsize, family="Times New Roman")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw YAMNet architecture diagram with Times New Roman font and adjustable fontsize")
    parser.add_argument("--fontsize", type=int, default=16, help="Base font size to use for text")
    parser.add_argument("--out", type=str, default="yamnet_diagram.png", help="Output filename (png/pdf supported)")
    args = parser.parse_args()
    draw_diagram(fontsize=args.fontsize, out_file=args.out)
