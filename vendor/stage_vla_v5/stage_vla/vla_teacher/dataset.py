"""
M13 Teacher Dataset definition.

Sample:
image + instruction + teacher_action + stage_label
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class TeacherSample:
    image_path: str
    instruction: str
    teacher_action: list
    stage_label: list


class TeacherDatasetBuilder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(asdict(sample))

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "samples.json", "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)
