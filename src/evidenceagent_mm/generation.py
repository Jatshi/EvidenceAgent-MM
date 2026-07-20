"""Optional evidence-constrained Qwen generator."""

from __future__ import annotations

from typing import Any

from evidenceagent_mm.schema import EvidenceAtom


class QwenEvidenceGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        device_map: str = "auto",
        max_new_tokens: int = 180,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install evidenceagent-mm[gpu] for Qwen generation") from exc
        self.torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, evidence: list[EvidenceAtom]) -> str:
        evidence_lines = [
            f"[{atom.evidence_id}] {atom.start_ms}-{atom.end_ms}ms "
            f"speaker={atom.speaker_id or 'unknown'} page={atom.page_no or 'n/a'}: {atom.text}"
            for atom in evidence
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer only from the supplied evidence. Do not invent names, times, pages, "
                    "or conclusions. Keep bracketed evidence IDs in the answer."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nEvidence:\n" + "\n".join(evidence_lines),
            },
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[0][inputs.input_ids.shape[1] :]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()
