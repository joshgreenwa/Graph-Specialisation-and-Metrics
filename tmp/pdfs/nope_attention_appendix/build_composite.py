from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation


SOURCE_PDFS = [
    Path("/Users/joshgreen/Downloads/prevtok_attn.pdf"),
    Path("/Users/joshgreen/Downloads/prevtok_std.pdf"),
    Path("/Users/joshgreen/Downloads/fixedpos_attn.pdf"),
    Path("/Users/joshgreen/Downloads/fixedpos_std.pdf"),
    Path("/Users/joshgreen/Downloads/contentmatch_attn.pdf"),
    Path("/Users/joshgreen/Downloads/contentmatch_std.pdf"),
]

OUTPUT_PDF = Path(
    "/Users/joshgreen/Documents/Graph Specialisation and Metrics/output/pdf/"
    "nope_attention_patterns_appendix.pdf"
)

# A 4.75-inch-wide composite fits comfortably on a portrait appendix page when
# constrained by height in LaTeX. One uniform scale keeps every heatmap cell and
# label identically sized. Smaller gaps pair each mean with its SD; larger gaps
# separate the three tasks.
COMPOSITE_WIDTH_PT = 4.75 * 72.0
OUTER_PADDING_PT = 2.0
PANEL_GAPS_PT = [2.0, 8.0, 2.0, 8.0, 2.0]


def main() -> None:
    missing = [str(path) for path in SOURCE_PDFS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source PDFs: {missing}")

    readers = [PdfReader(path) for path in SOURCE_PDFS]
    pages = []
    for path, reader in zip(SOURCE_PDFS, readers):
        if len(reader.pages) != 1:
            raise ValueError(f"Expected a single-page PDF: {path}")
        pages.append(reader.pages[0])

    widest_source = max(float(page.mediabox.width) for page in pages)
    scale = COMPOSITE_WIDTH_PT / widest_source
    panel_heights = [float(page.mediabox.height) * scale for page in pages]

    page_height = (
        2.0 * OUTER_PADDING_PT + sum(panel_heights) + sum(PANEL_GAPS_PT)
    )
    writer = PdfWriter()
    canvas = writer.add_blank_page(width=COMPOSITE_WIDTH_PT, height=page_height)

    y_top = page_height - OUTER_PADDING_PT
    for index, (page, panel_height) in enumerate(zip(pages, panel_heights)):
        y_top -= panel_height
        scaled_width = float(page.mediabox.width) * scale
        x_offset = (COMPOSITE_WIDTH_PT - scaled_width) / 2.0
        transform = Transformation().scale(scale).translate(tx=x_offset, ty=y_top)
        canvas.merge_transformed_page(page, transform, expand=False)
        if index < len(PANEL_GAPS_PT):
            y_top -= PANEL_GAPS_PT[index]

    canvas.compress_content_streams()

    writer.add_metadata(
        {
            "/Title": "Attention patterns for synthetic lookup tasks",
            "/Subject": "Mean attention and across-input attention variability by layer",
            "/Creator": "Codex composite from publication-quality vector source PDFs",
        }
    )
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PDF.open("wb") as stream:
        writer.write(stream)

    print(OUTPUT_PDF)
    print(f"page_size_pt={COMPOSITE_WIDTH_PT:.3f}x{page_height:.3f}")


if __name__ == "__main__":
    main()
