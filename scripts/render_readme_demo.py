"""Render the deterministic README demo GIF from verified release facts."""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - documentation utility
    raise SystemExit("Install Pillow first: python -m pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "evidenceagent_v2_demo.gif"
SIZE = (960, 540)
BG = "#090B16"
PANEL = "#12162A"
MUTED = "#8F9AB8"
WHITE = "#F7F8FF"
PURPLE = "#9B7BFF"
CYAN = "#58E6D9"
AMBER = "#FFCF67"
RED = "#FF7285"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: str = WHITE,
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, accent: str
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#252B49", width=2)
    x1, y1, _, _ = box
    draw.rounded_rectangle((x1 + 16, y1 + 16, x1 + 22, y1 + 54), radius=3, fill=accent)
    text(draw, (x1 + 36, y1 + 16), title.upper(), 14, MUTED, True)
    text(draw, (x1 + 36, y1 + 43), value, 22, WHITE, True)


def base(step: int, label: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    for x in range(0, SIZE[0], 32):
        for y in range(0, SIZE[1], 32):
            draw.ellipse((x, y, x + 2, y + 2), fill="#171B31")
    text(draw, (42, 28), "EVIDENCEAGENT-MM", 24, WHITE, True)
    text(draw, (42, 60), "VERIFIABLE MULTIMODAL AGENT", 13, PURPLE, True)
    text(draw, (750, 34), "v2.0.0", 16, CYAN, True)
    text(draw, (42, 106), f"0{step}  {label}", 18, MUTED, True)
    draw.rounded_rectangle((42, 500, 918, 505), radius=3, fill="#202641")
    draw.rounded_rectangle((42, 500, 42 + int(876 * step / 6), 505), radius=3, fill=PURPLE)
    return image, draw


def render() -> list[Image.Image]:
    frames: list[Image.Image] = []
    stages = [
        (
            "INGEST",
            [
                ("AUDIO", "ASR + timestamps", CYAN),
                ("SCREEN", "OCR + page IDs", PURPLE),
                ("TURNS", "speaker segments", AMBER),
            ],
        ),
        (
            "NORMALIZE",
            [
                ("EVIDENCE ATOMS", "typed + replayable", PURPLE),
                ("PROVENANCE", "speaker / time / page", CYAN),
                ("HASH", "stable identity", AMBER),
            ],
        ),
        (
            "RETRIEVE",
            [
                ("FTS5", "lexical candidates", AMBER),
                ("BGE-M3", "semantic candidates", CYAN),
                ("RRF + GRAPH", "bounded fusion", PURPLE),
            ],
        ),
        (
            "VERIFY",
            [
                ("AMBIGUITY", "clarify if unresolved", AMBER),
                ("SUPPORT", "claim-level evidence", CYAN),
                ("VISUAL GATE", "page claim needs OCR", PURPLE),
            ],
        ),
        (
            "ANSWER",
            [
                ("STATUS", "answered", CYAN),
                ("CITATIONS", "speaker + 00:42 + p.7", PURPLE),
                ("CONTRACT SCORE", "0.920 test", AMBER),
            ],
        ),
        (
            "FAIL CLOSED",
            [
                ("STATUS", "abstained", RED),
                ("MISSING", "required screen evidence", AMBER),
                ("NEXT ACTION", "ask targeted question", CYAN),
            ],
        ),
    ]
    for index, (label, cards) in enumerate(stages, start=1):
        image, draw = base(index, label)
        text(
            draw,
            (42, 145),
            "Who proposed the rollback plan, when, and on which slide?",
            22,
            WHITE,
            True,
        )
        for card_index, (title, value, accent) in enumerate(cards):
            left = 42 + card_index * 292
            card(draw, (left, 214, left + 268, 326), title, value, accent)
        verdict = (
            "Every claim is replayable"
            if index == 5
            else "No evidence, no answer"
            if index == 6
            else "Evidence stays structured"
        )
        draw.rounded_rectangle(
            (42, 365, 918, 460), radius=20, fill="#0E2130", outline="#1D5960", width=2
        )
        text(draw, (68, 389), verdict, 25, CYAN, True)
        text(
            draw, (68, 426), "Qwen3-1.7B LoRA  |  SFT -> DPO -> GRPO  |  single RTX 4090", 15, MUTED
        )
        frames.append(image)
    return frames


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = render()
    frames[0].save(
        OUTPUT, save_all=True, append_images=frames[1:], duration=1150, loop=0, optimize=True
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
