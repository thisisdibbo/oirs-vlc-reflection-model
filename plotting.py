"""
Figure saving that survives a PDF viewer holding the file open.

On Windows, Adobe Reader and Edge lock an open PDF, and matplotlib's savefig
raises OSError -- PermissionError on some paths, [Errno 22] Invalid argument on
others. That kills a script AFTER the expensive run has finished, which during
a sprint is the worst possible time. save_fig falls back to a timestamped
filename and never raises.

The PDF is the copy that ships: it is vector, so it stays sharp at any size in
the typeset paper. The PNG is a convenience copy for quick viewing, rendered at
freeze.FIG_DPI.
"""

import time

_TS = time.strftime("%H%M%S")

try:
    from freeze import FIG_DPI as _DPI
except Exception:            # allow the module to be used stand-alone
    _DPI = 300


def publication_style():
    """
    Consistent typography across every figure in the paper.

    Called once at import by the day scripts that want it. Serif to match the
    body text of an Elsevier article.

    The sizes below assume every figure is authored at its FINAL physical size
    -- about 6.5 in wide, the text width of an elsarticle preprint on letter
    paper -- so that \\includegraphics[width=\\textwidth] scales it by roughly
    1.0 and these are the point sizes that actually reach the page. Authoring
    wider and letting LaTeX shrink the figure is what put 3.5 pt tick labels in
    an earlier draft. Do not raise any figsize without raising these to match.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.size":          8,
        "axes.titlesize":     8.5,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    6.5,
        "figure.titlesize":   9,
        "axes.grid":          True,
        "grid.alpha":         0.30,
        "grid.linewidth":     0.4,
        "lines.linewidth":    1.1,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "0.8",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,     # embed TrueType, not Type 3
        "ps.fonttype":        42,
    })


def save_fig(fig, stem, formats=("pdf", "png"), dpi=None):
    """
    Save `fig` as <stem>.<fmt>, falling back to <stem>_<HHMMSS>.<fmt> if the
    target is locked. Returns the list of filenames actually written; never
    raises, because losing a figure is recoverable and losing the run is not.
    """
    dpi = _DPI if dpi is None else dpi
    written = []
    for fmt in formats:
        for name in (f"{stem}.{fmt}", f"{stem}_{_TS}.{fmt}"):
            try:
                fig.savefig(name, dpi=None if fmt == "pdf" else dpi)
                written.append(name)
                break
            except OSError as exc:
                print(f"  [locked] {name}: {exc.__class__.__name__} -- "
                      f"writing a timestamped copy instead")
            except Exception as exc:          # never kill a finished run
                print(f"  [failed] {name}: {exc.__class__.__name__}: {exc}")
                break
    return written
