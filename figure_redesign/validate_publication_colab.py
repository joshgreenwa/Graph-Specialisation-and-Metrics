"""Render and structurally validate every publication figure in the Colab file."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader


ROOT = Path("/Users/joshgreen/Documents/Graph Specialisation and Metrics")
NOTEBOOK = ROOT / "output/notebooks/dissertation_final_mixed_synthetic_publication.ipynb"
QA_ROOT = ROOT / "tmp/pdfs/publication_notebook_qa"
FIGURE_DIR = QA_ROOT / "figures"
RENDER_DIR = QA_ROOT / "rendered"
CONTACT_SHEET = QA_ROOT / "publication_notebook_contact_sheet.png"


EXPECTED_SIZES = {
    "fig1a_raw_task_scores": (3.315, 3.12),
    "fig1b_sensitivity_selectivity": (3.315, 3.12),
    "fig2a_sensitivity_impact": (2.1766666667, 2.36),
    "fig2b_selectivity_ablation": (2.1766666667, 2.36),
    "fig2c_selectivity_rescue": (2.1766666667, 2.36),
    "fig3_joint_ablation_tests": (6.85, 4.25),
    "fig4a_necessity": (3.315, 2.70),
    "fig4b_rescue": (3.315, 2.70),
    "fig5_attention_visualisations": (6.85, 4.75),
}


def load_notebook_module() -> types.ModuleType:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != 1:
        raise AssertionError("notebook must contain one cell")
    cell = notebook["cells"][0]
    if cell.get("outputs") or cell.get("execution_count") is not None:
        raise AssertionError("delivered notebook must not contain stale outputs")
    source = "".join(cell["source"])
    compile(source, str(NOTEBOOK), "exec")
    module = types.ModuleType("publication_colab_qa")
    sys.modules[module.__name__] = module
    exec(compile(source, str(NOTEBOOK), "exec"), module.__dict__)
    return module


def mock_results(module: types.ModuleType, cfg: object) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(705)
    results: list[dict] = []
    rows: list[dict] = []
    families = (
        "semantic_specialist",
        "structural_specialist",
        "high_J_generalist",
        "low_J_inert",
    )
    terminals = {
        "semantic_specialist": (1.12, 0.24),
        "structural_specialist": (0.25, 1.08),
        "high_J_generalist": (0.72, 0.78),
        "low_J_inert": (0.12, 0.10),
    }

    for seed_index, seed in enumerate(cfg.seeds):
        semantic = np.empty((cfg.layers, cfg.heads), dtype=float)
        structural = np.empty_like(semantic)
        for layer in range(cfg.layers):
            for head in range(cfg.heads):
                phase = 2.0 * np.pi * head / cfg.heads
                strength = np.exp(-1.35 + 0.42 * layer + rng.normal(0.0, 0.18))
                selectivity = np.clip(
                    0.75 * np.sin(phase + 0.48 * layer) + rng.normal(0.0, 0.12),
                    -0.94,
                    0.94,
                )
                semantic[layer, head] = strength * (1.0 + selectivity)
                structural[layer, head] = strength * (1.0 - selectivity)

        # Mirror the isolated lower-tail structural failure seen in the real
        # output so the display-only exclusion and zoom are regression-tested.
        if seed_index == 0:
            structural[0, 0] = 1.0e-8
            semantic[0, 0] = 4.0e-2
        elif seed_index == 1:
            structural[1, 1] = 3.0e-1
            semantic[1, 1] = 5.0e-5

        selected_semantic = [(2, 1), (2, 2), (1, 1)]
        selected_structural = [(2, 5), (2, 6), (1, 6)]
        family_payload: dict[str, dict] = {}
        for task_index, task in enumerate(("semantic", "structural")):
            family_payload[task] = {"families": {}}
            for family_index, family in enumerate(families):
                terminal = terminals[family][task_index]
                x_values = np.arange(4, dtype=float)
                mean = terminal * (x_values / 3.0) ** 1.15
                functional = np.column_stack((
                    mean * (0.96 + 0.02 * seed_index),
                    mean * (1.02 + 0.01 * family_index),
                ))
                accuracy = np.column_stack((
                    0.045 * mean * (0.94 + 0.02 * seed_index),
                    0.045 * mean * (1.03 + 0.01 * family_index),
                ))
                family_payload[task]["families"][family] = {
                    "functional": functional,
                    "loss": 0.35 * functional,
                    "accuracy_drop": accuracy,
                    "head_order": [(index // cfg.heads, index % cfg.heads) for index in range(3)],
                }

        results.append({
            "seed": int(seed),
            "semantic_score": semantic,
            "structural_score": structural,
            "selected_groups": {
                "semantic": selected_semantic,
                "structural": selected_structural,
            },
            "dj_family_ablation": {
                "revision": module.DJ_FAMILY_ABLATION_REVISION,
                "tasks": family_payload,
            },
        })

        semantic_norm = semantic / semantic.mean()
        structural_norm = structural / structural.mean()
        joint = 0.5 * (semantic_norm + structural_norm)
        selectivity = (semantic_norm - structural_norm) / (semantic_norm + structural_norm)
        for layer in range(cfg.layers):
            for head in range(cfg.heads):
                j_value = float(joint[layer, head])
                d_value = float(selectivity[layer, head])
                impact = max(0.025, 0.36 * j_value ** 0.9 * np.exp(rng.normal(0.0, 0.22)))
                ablation_contrast = float(np.clip(0.80 * d_value + rng.normal(0.0, 0.13), -1, 1))
                rescue_contrast = float(0.18 * d_value + rng.normal(0.0, 0.035))
                rows.append({
                    "seed": int(seed),
                    "layer": layer,
                    "head": head,
                    "joint_score_J": j_value,
                    "selectivity_D": d_value,
                    "selectivity_reliable": bool(j_value >= module.DJ_RELIABILITY_FLOOR),
                    "DJ_class": module.dj_class(j_value, d_value),
                    "ablation_joint_impact": impact,
                    "ablation_role_selectivity": ablation_contrast,
                    "semantic_ablation_functional_norm": impact * (1.0 + ablation_contrast),
                    "structural_ablation_functional_norm": impact * (1.0 - ablation_contrast),
                    "semantic_rescue_mem": 0.12 + 0.5 * rescue_contrast,
                    "structural_rescue_mem": 0.12 - 0.5 * rescue_contrast,
                    "rescue_role_contrast": rescue_contrast,
                })
    return results, rows


def mock_attention(cfg: object) -> dict:
    rng = np.random.default_rng(918)
    specifications = (
        ("Semantic specialist", 10, 11, 3, 7),
        ("Structural specialist", 12, 2, 3, 5),
        ("High-J generalist", 3, 4, 2, 4),
        ("First-layer head", 1, 13, 1, 8),
    )
    examples = []
    for role, query, source, layer_display, head_display in specifications:
        matrix = rng.uniform(0.005, 0.08, size=(cfg.n, cfg.n))
        matrix[source, :] += 0.28
        matrix[(source - 1) % cfg.n, :] += 0.08
        matrix[(source + 1) % cfg.n, :] += 0.08
        matrix /= matrix.sum(axis=0, keepdims=True)
        examples.append({
            "role": role,
            "query_node": query,
            "source_node": source,
            "layer_display": layer_display,
            "head_display": head_display,
            "joint_score_J": 1.2,
            "selectivity_D_rel": 0.4,
            "attention_matrix": matrix,
            "query_attention": matrix[:, query],
        })
    return {"examples": examples}


def render_figures(module: types.ModuleType) -> list[Path]:
    if QA_ROOT.exists():
        shutil.rmtree(QA_ROOT)
    FIGURE_DIR.mkdir(parents=True)
    cfg = module.Config()
    results, rows = mock_results(module, cfg)
    structural_cube = np.stack([result["structural_score"] for result in results])
    semantic_cube = np.stack([result["semantic_score"] for result in results])
    raw_keep = module.raw_score_display_mask(structural_cube, semantic_cube)
    if (
        int((~raw_keep).sum()) != 2
        or bool(raw_keep[0, 0, 0])
        or bool(raw_keep[1, 1, 1])
    ):
        raise AssertionError("raw-score display mask must omit every sub-1e-4 channel")
    paths: list[str] = []
    paths.extend(module.figure_specialisation_plane(results, cfg, FIGURE_DIR))
    paths.extend(module.figure_joint_selectivity_plane(rows, cfg, FIGURE_DIR))
    causal, summary, quadrant_rows = module.figure_joint_selectivity_validation(
        rows,
        cfg,
        FIGURE_DIR,
    )
    paths.extend(causal)
    if len(quadrant_rows) != 12:
        raise AssertionError("quadrant summary schema changed")
    for key in (
        "J_vs_joint_ablation",
        "D_vs_ablation_role",
        "D_vs_ablation_role_all_heads",
        "D_vs_rescue_role",
        "D_vs_rescue_role_all_heads",
    ):
        if key not in summary:
            raise AssertionError(f"missing causal summary key: {key}")
    ablation, _ = module.figure_dj_family_ablation(results, cfg, FIGURE_DIR)
    paths.extend(ablation)
    paths.extend(module.figure_necessity_matrix(
        {"necessity_mean": np.array([[0.004, 0.052], [-0.001, 0.011]])},
        FIGURE_DIR,
    ))
    paths.extend(module.figure_rescue_matrix(
        {"rescue_mean": np.array([[0.154, 0.066], [-0.008, 0.128]])},
        FIGURE_DIR,
    ))
    paths.extend(module.figure_attention_visualisations(mock_attention(cfg), cfg, FIGURE_DIR))
    plt.close("all")
    output_paths = [Path(path) for path in paths]
    if len(output_paths) != 18 or any(not path.exists() for path in output_paths):
        raise AssertionError("expected nine complete PNG/PDF pairs")
    return output_paths


def walk_resources(value: object):
    if hasattr(value, "get_object"):
        value = value.get_object()
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_resources(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from walk_resources(child)


def inspect_pdfs(paths: list[Path]) -> None:
    pngs = sorted((path for path in paths if path.suffix == ".png"), key=lambda path: path.stem)
    for png in pngs:
        expected_width, expected_height = EXPECTED_SIZES[png.stem]
        expected_pixels = (
            round(expected_width * 600),
            round(expected_height * 600),
        )
        actual_pixels = Image.open(png).size
        if any(
            abs(actual - expected) > 1
            for actual, expected in zip(actual_pixels, expected_pixels)
        ):
            raise AssertionError(
                f"{png.name} has {actual_pixels} pixels, expected {expected_pixels}"
            )
        print(f"[png] {png.name}: {actual_pixels[0]} x {actual_pixels[1]} px at 600 ppi")

    pdfs = sorted((path for path in paths if path.suffix == ".pdf"), key=lambda path: path.stem)
    if [path.stem for path in pdfs] != sorted(EXPECTED_SIZES):
        raise AssertionError("unexpected output stems")
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        if len(reader.pages) != 1:
            raise AssertionError(f"{pdf.name} must contain one page")
        page = reader.pages[0]
        width = float(page.mediabox.width) / 72.0
        height = float(page.mediabox.height) / 72.0
        expected_width, expected_height = EXPECTED_SIZES[pdf.stem]
        if abs(width - expected_width) > 0.002 or abs(height - expected_height) > 0.002:
            raise AssertionError(
                f"{pdf.name} has {width:.4f} x {height:.4f} in, expected "
                f"{expected_width:.4f} x {expected_height:.4f} in"
            )
        resources = page.get("/Resources", {})
        type3_fonts = []
        images = []
        for resource in walk_resources(resources):
            if resource.get("/Subtype") == "/Type3":
                type3_fonts.append(resource)
            if resource.get("/Subtype") == "/Image":
                images.append((int(resource.get("/Width", 0)), int(resource.get("/Height", 0))))
        if type3_fonts:
            raise AssertionError(f"{pdf.name} contains Type 3 fonts")
        # The intentionally rasterised colourbar gradient is long and thin at
        # exactly 600 ppi. No square-ish image may replace matrix or plot data.
        if any(width <= 10 * max(height, 1) for width, height in images):
            raise AssertionError(f"{pdf.name} contains a non-colourbar raster object: {images}")
        print(
            f"[pdf] {pdf.name}: {width:.4f} x {height:.4f} in; "
            f"Type3=0; image XObjects={images}"
        )


def render_with_poppler(paths: list[Path]) -> list[Path]:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    for pdf in sorted(path for path in paths if path.suffix == ".pdf"):
        prefix = RENDER_DIR / pdf.stem
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "180", str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rendered.append(prefix.with_suffix(".png"))
    return rendered


def make_contact_sheet(rendered: list[Path]) -> None:
    full_width = 1500
    pair_gap = 36
    row_gap = 28
    background = "#E8EBEE"
    rows = (
        (rendered[0], rendered[1]),
        (rendered[2], rendered[3], rendered[4]),
        (rendered[5],),
        (rendered[6], rendered[7]),
        (rendered[8],),
    )
    row_images = []
    for row in rows:
        target_width = (full_width - pair_gap * (len(row) - 1)) // len(row)
        images = []
        for path in row:
            image = Image.open(path).convert("RGB")
            image.thumbnail((target_width, 2000), Image.Resampling.LANCZOS)
            images.append(image)
        canvas = Image.new("RGB", (full_width, max(image.height for image in images)), "white")
        left = 0
        for image in images:
            canvas.paste(image, (left, 0))
            left += target_width + pair_gap
        row_images.append(canvas)
    sheet = Image.new(
        "RGB",
        (full_width, sum(image.height for image in row_images) + row_gap * (len(row_images) - 1)),
        background,
    )
    top = 0
    for image in row_images:
        sheet.paste(image, (0, top))
        top += image.height + row_gap
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width - 1, sheet.height - 1), outline="#C9CED4", width=2)
    sheet.save(CONTACT_SHEET, dpi=(180, 180))


def main() -> None:
    module = load_notebook_module()
    paths = render_figures(module)
    inspect_pdfs(paths)
    rendered = render_with_poppler(paths)
    make_contact_sheet(rendered)
    print(f"[qa] contact sheet: {CONTACT_SHEET}")


if __name__ == "__main__":
    main()
