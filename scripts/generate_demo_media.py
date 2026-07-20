"""Generate a redistributable two-speaker meeting video with slide changes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/demo_meeting")
    args = parser.parse_args()
    for binary in ("ffmpeg", "espeak-ng"):
        if shutil.which(binary) is None:
            raise SystemExit(f"required executable not found: {binary}")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    speech = [
        (
            "en-us+m3",
            "I propose design B because it reduces retrieval latency to forty two milliseconds.",
        ),
        ("en-us+f3", "I suggest we review the budget and delivery risk next week."),
    ]
    for index, (voice, text) in enumerate(speech, 1):
        run(["espeak-ng", "-v", voice, "-s", "135", "-w", str(root / f"speaker-{index}.wav"), text])
    speech_durations = [
        float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(root / f"speaker-{index}.wav"),
                ],
                text=True,
            ).strip()
        )
        for index in (1, 2)
    ]
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(root / "speaker-1.wav"),
            "-f",
            "lavfi",
            "-t",
            "1.5",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-i",
            str(root / "speaker-2.wav"),
            "-filter_complex",
            "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]",
            "-map",
            "[a]",
            str(root / "meeting.wav"),
        ]
    )
    slide_specs = [
        (1, "DESIGN REVIEW", "Design B / P95 latency: 42 ms", "0x132a45"),
        (2, "NEXT STEPS", "Budget and delivery risk / next week", "0x253c2c"),
    ]
    for page, title, subtitle, color in slide_specs:
        draw = (
            f"drawtext=fontfile={font}:text='{title}':fontcolor=white:fontsize=62:x=80:y=110,"
            f"drawtext=fontfile={font}:text='{subtitle}':fontcolor=0x4ce3d4:fontsize=38:x=80:y=230,"
            f"drawtext=fontfile={font}:text='PAGE {page}':fontcolor=white:fontsize=24:x=w-tw-50:y=h-th-40"
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=1280x720:d=1",
                "-vf",
                draw,
                "-frames:v",
                "1",
                str(root / f"slide-{page}.png"),
            ]
        )
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(root / "meeting.wav"),
            ],
            text=True,
        ).strip()
    )
    first_duration = duration * 0.58
    concat_file = root / "slides.txt"
    concat_file.write_text(
        f"file '{(root / 'slide-1.png').resolve()}'\nduration {first_duration:.3f}\n"
        f"file '{(root / 'slide-2.png').resolve()}'\nduration {duration - first_duration:.3f}\n"
        f"file '{(root / 'slide-2.png').resolve()}'\n",
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(root / "meeting.wav"),
            "-shortest",
            "-vf",
            "fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(root / "meeting.mp4"),
        ]
    )
    manifest = {
        "license": "CC0-1.0",
        "synthetic": True,
        "duration_seconds": duration,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "transcript": [item[1] for item in speech],
        "slide_change_seconds": first_duration,
        "gold_turns": [
            {"speaker_id": "SPEAKER_00", "start_seconds": 0.0, "end_seconds": speech_durations[0]},
            {
                "speaker_id": "SPEAKER_01",
                "start_seconds": speech_durations[0] + 1.5,
                "end_seconds": speech_durations[0] + 1.5 + speech_durations[1],
            },
        ],
        "generator": {"speech": "eSpeak-NG", "media": "FFmpeg"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
