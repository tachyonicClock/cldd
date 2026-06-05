#!/usr/bin/env python3
"""Render LaTeX tables to PNG files.

This script expects input .tex files to contain LaTeX table content (for example a
`tabular` environment). For each matching file, it builds a small standalone LaTeX
document, compiles it to PDF, then converts that PDF to PNG.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEX_TEMPLATE = r"""\documentclass[border=%dpt]{standalone}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\begin{document}
\small
\input{%s}
\end{document}
"""


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        output = proc.stdout.strip()
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}"
        )


def convert_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int) -> None:
    pdftoppm = shutil.which("pdftoppm")
    magick = shutil.which("magick")

    if pdftoppm:
        # pdftoppm writes to <prefix>.png, so pass the path without suffix.
        prefix = png_path.with_suffix("")
        run(
            [
                pdftoppm,
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ]
        )
        return

    if magick:
        run(
            [
                magick,
                "-density",
                str(dpi),
                str(pdf_path),
                "-quality",
                "100",
                str(png_path),
            ]
        )
        return

    raise RuntimeError(
        "No PDF-to-PNG converter found. Install either 'pdftoppm' (poppler) or "
        "ImageMagick ('magick')."
    )


def render_table(
    tex_file: Path,
    output_dir: Path,
    dpi: int,
    keep_temp: bool,
    border_pt: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{tex_file.stem}.png"

    with tempfile.TemporaryDirectory(prefix=f"render_{tex_file.stem}_") as tmp:
        tmpdir = Path(tmp)
        local_tex = tmpdir / tex_file.name
        wrapper_tex = tmpdir / "wrapper.tex"

        local_tex.write_text(tex_file.read_text(encoding="utf-8"), encoding="utf-8")
        wrapper_tex.write_text(
            TEX_TEMPLATE % (border_pt, tex_file.name),
            encoding="utf-8",
        )

        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            raise RuntimeError(
                "'pdflatex' was not found in PATH. Install a LaTeX distribution "
                "(e.g., TeX Live)."
            )

        run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tmpdir),
                str(wrapper_tex),
            ],
            cwd=tmpdir,
        )

        pdf_path = tmpdir / "wrapper.pdf"
        convert_pdf_to_png(pdf_path, out_png, dpi)

        if keep_temp:
            keep_dir = output_dir / f"_tmp_{tex_file.stem}"
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(tmpdir, keep_dir)

    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX tables to PNG images")
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("table"),
        help="Directory containing input .tex table files (default: table)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("table"),
        help="Directory for output PNG files (default: table)",
    )
    parser.add_argument(
        "--pattern",
        default="*.tex",
        help="Glob pattern for selecting table files (default: *.tex)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in DPI (default: 300)",
    )
    parser.add_argument(
        "--border-pt",
        type=int,
        default=18,
        help="Padding around table in output PDF, in points (default: 18)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary build files in output directory for debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    table_dir = args.table_dir
    if not table_dir.exists() or not table_dir.is_dir():
        print(f"error: table directory not found: {table_dir}", file=sys.stderr)
        return 2

    tex_files = sorted(table_dir.glob(args.pattern))
    if not tex_files:
        print(
            f"error: no files matching pattern '{args.pattern}' in {table_dir}",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for tex_file in tex_files:
        try:
            out_png = render_table(
                tex_file=tex_file,
                output_dir=args.output_dir,
                dpi=args.dpi,
                keep_temp=args.keep_temp,
                border_pt=args.border_pt,
            )
            print(f"ok: {tex_file} -> {out_png}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"error: {tex_file}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
