# Model and system card

## EvidenceAgent-MM 2.0 adapter

The public trainable artifact is
[jatshi/EvidenceAgent-MM-Qwen3-1.7B-GRPO-LoRA](https://huggingface.co/jatshi/EvidenceAgent-MM-Qwen3-1.7B-GRPO-LoRA).
It is a PEFT LoRA adapter over `Qwen/Qwen3-1.7B`, produced by the committed
SFT -> DPO -> GRPO pipeline on one RTX 4090. The final GRPO stage ran for 100
optimizer steps; mean shaped reward was 0.7101, rising from 0.5532 over the
first 20 steps to 0.7796 over the final 20 steps.

The 120-question synthetic benchmark uses session-level splits
(8 train sessions/80 examples, 2 validation/20, 2 test/20). Both validation
and test achieved a 0.920 composite contract score, with 1.000 valid-JSON and
grounding scores, 0.800 citation score, and 0.800 abstention score. Test mean
generation latency was 5.383 seconds, P95 was 6.281 seconds, and peak evaluation
VRAM was 3.65 GiB.

These figures validate the repository's output contract and evidence-control
logic on a small synthetic benchmark. They do not establish real-meeting
accuracy, semantic entailment, confidence calibration, speaker recognition,
or broad domain generalization.

## System behavior

EvidenceAgent-MM retrieves typed evidence, expands a bounded graph, verifies modality and support requirements, and returns one of three states. The default renderer is deterministic. Qwen3 generation is optional and runs only after evidence selection and gating.

The historical v0.1 system-level repository remains available at
<https://huggingface.co/jatshi/EvidenceAgent-MM>. That release did not train a
new neural checkpoint and does not relabel upstream weights. The separate v2.0
repository above contains only the trained adapter and its reproducibility
metadata; users must obtain the attributed Qwen base model upstream.

## Model inventory

| Component | Default/reference | Role | License source |
|---|---|---|---|
| ASR | faster-whisper small smoke; large-v3-turbo target | timestamped transcript | upstream repository/model card |
| Diarization | pyannote Community-1 | anonymous speaker turns | gated upstream model card |
| OCR | PaddleOCR 3.7, PP-OCRv5 mobile det/rec smoke | slide/screen text | upstream repository/model card |
| Embedding | hashing baseline; BGE-M3 production | multilingual retrieval | BGE-M3 model card |
| Generation | deterministic baseline; Qwen3-8B optional | evidence-constrained wording | Qwen3-8B model card |

Exact revisions, packages, GPU, driver, elapsed time, and output hashes belong in each integration report. A model name alone is not sufficient provenance.

The v0.1 integration reports verify that each optional adapter executes on the target RTX 4090 and preserves the evidence contract. They do not establish accuracy on real meetings. In the synthetic smoke, ASR makes one lexical error, mobile OCR omits one numeric phrase, and the ungated diarization fallback detects turns rather than persistent speakers. These observed errors are retained in the published JSON rather than corrected by hand.

## Limitations

- Hashing retrieval is a reproducible baseline, not a semantic retrieval ceiling.
- The initial confidence formula is not calibrated and has poor ECE on Bronze.
- Citation presence does not by itself prove semantic entailment; claim support needs separate evaluation.
- Overlapping speech, accented speech, low-resolution screen sharing, and rapid slide changes can break upstream perception.
- Anonymous speaker IDs are not identities and must not be presented as real-person recognition.

## Safety

Use access control and retention limits for meeting media. Do not index confidential meetings in a public Demo. Do not treat a generated answer as an official record without replaying its citations.
