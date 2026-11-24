#!/usr/bin/env python3
"""R2 spike: parse one OMG PDF with Docling (OCR off for born-digital specs)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
OUT_DIR = ROOT / "data" / "parsed"


def main() -> int:
    if len(sys.argv) < 2:
        console.print("[red]Usage: python scripts/parse_spike.py <pdf-filename>[/red]")
        return 1

    pdf_name = sys.argv[1]
    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (pdf_path.stem + ".json")

    # Born-digital PDFs → no OCR, faster and cleaner
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    console.print(f"[cyan]Parsing[/cyan] {pdf_path.name} (OCR=off) …")
    t0 = time.perf_counter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    elapsed = time.perf_counter() - t0
    console.print(f"[green]Done in {elapsed:.1f}s[/green]")

    data = {
        "source": pdf_path.name,
        "elapsed_seconds": round(elapsed, 2),
        "pages": len(doc.pages) if hasattr(doc, "pages") else None,
        "texts_count": len(doc.texts) if hasattr(doc, "texts") else None,
        "tables_count": len(doc.tables) if hasattr(doc, "tables") else None,
        "markdown_preview": doc.export_to_markdown()[:4000],
        "export": doc.export_to_dict(),
    }

    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    console.print(f"Wrote [cyan]{out_path}[/cyan]")
    console.print(
        f"Pages: {data['pages']}  |  texts: {data['texts_count']}  |  tables: {data['tables_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
