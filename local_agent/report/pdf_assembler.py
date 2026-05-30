"""Assemble PNG figures + markdown sections into a single PDF report.

Uses weasyprint (HTML -> PDF) since pandoc/xelatex aren't available on this HPC.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import markdown as md_lib

# Defer weasyprint import until use to keep `--no-pdf` paths cheap.


REPORT_CSS = """
@page {
    size: A4;
    margin: 18mm 16mm;
    @bottom-center {
        content: "EnSEMBLE Report — page " counter(page);
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: "DejaVu Sans", "Helvetica", sans-serif;
    font-size: 10pt;
    color: #222;
    line-height: 1.35;
}
h1 {
    font-size: 18pt;
    color: #1a3a5c;
    margin-bottom: 0.1em;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 4px;
}
h2 {
    font-size: 13pt;
    color: #1a3a5c;
    margin-top: 1.4em;
    margin-bottom: 0.4em;
}
h3 {
    font-size: 11pt;
    color: #2c4f70;
    margin-top: 1.0em;
    margin-bottom: 0.3em;
}
.meta {
    color: #666;
    font-size: 9pt;
    margin-bottom: 1.5em;
}
.figure {
    margin: 1em 0;
    page-break-inside: avoid;
    text-align: center;
}
.figure img {
    max-width: 100%;
    height: auto;
}
.figure-caption {
    font-size: 9pt;
    color: #555;
    margin-top: 0.3em;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 8pt;
    margin: 0.5em 0;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 6px;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #f0f3f6;
    color: #1a3a5c;
    font-weight: bold;
}
strong { color: #1a3a5c; }
.footer {
    margin-top: 2em;
    padding-top: 0.5em;
    border-top: 1px solid #ddd;
    color: #666;
    font-size: 8pt;
}
"""


def render_html_report(
    dataset_id: str,
    figure_paths: dict,  # {"compression": Path, "network": Path, "helper": Path}
    mini_thesis_md: Optional[str],
    verdict_table_md: str,
    parameters_line: str,
) -> str:
    today = datetime.date.today().isoformat()

    md_extensions = ["tables", "fenced_code"]

    thesis_html = (
        md_lib.markdown(mini_thesis_md, extensions=md_extensions)
        if mini_thesis_md
        else "<p><em>Mini-thesis not yet generated for this dataset.</em></p>"
    )
    table_html = md_lib.markdown(verdict_table_md, extensions=md_extensions)

    def fig_block(label: str, path: Path, caption: str) -> str:
        if not path or not Path(path).exists():
            return f'<div class="figure"><em>(figure not generated: {label})</em></div>'
        # Use file:// URI so weasyprint can resolve it
        uri = Path(path).resolve().as_uri()
        return f"""
        <div class="figure">
            <img src="{uri}" alt="{label}"/>
            <div class="figure-caption">{caption}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <title>EnSEMBLE Report: {dataset_id}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <h1>EnSEMBLE Report: {dataset_id}</h1>
    <div class="meta">Generated: {today}</div>

    <h2>Figure 1 — Compression Summary</h2>
    {fig_block("compression", figure_paths.get("compression"), "Funnel: significant gene sets &rarr; clustered themes &rarr; final verdicts.")}

    <h2>Figure 2 — Evidence Network</h2>
    {fig_block("network", figure_paths.get("network"), "Bipartite helper &harr; theme links (SUPPORTED + PARTIAL only). GENE_LEVEL_ONLY themes are excluded.")}

    <h2>Figure 3 — ESEA Helper Overview</h2>
    {fig_block("helper", figure_paths.get("helper"), "All significant ESEA helpers. Filled dots = used in at least one SUPPORTED/PARTIAL verdict; open dots = unlinked.")}

    <h2>Mini-Thesis</h2>
    {thesis_html}

    <h2>Appendix: Full Verdict Table</h2>
    {table_html}

    <div class="footer">{parameters_line}</div>
</body>
</html>
"""
    return html


def html_to_pdf(html: str, output_path: Path) -> Path:
    """Render HTML to PDF via weasyprint."""
    from weasyprint import HTML

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(output_path))
    return output_path


def assemble_report(
    dataset_id: str,
    figures_dir: Path,
    mini_thesis_path: Optional[Path],
    verdict_table_md_path: Path,
    output_pdf_path: Path,
    parameters_line: str = "",
) -> Path:
    figs = {
        "compression": figures_dir / "fig_compression.png",
        "network": figures_dir / "fig_network.png",
        "helper": figures_dir / "fig_esea_overview.png",
    }
    thesis_md = (
        Path(mini_thesis_path).read_text() if mini_thesis_path and Path(mini_thesis_path).exists() else None
    )
    table_md = Path(verdict_table_md_path).read_text()

    html = render_html_report(
        dataset_id=dataset_id,
        figure_paths=figs,
        mini_thesis_md=thesis_md,
        verdict_table_md=table_md,
        parameters_line=parameters_line,
    )
    return html_to_pdf(html, output_pdf_path)
