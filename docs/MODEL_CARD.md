# Model and system card

## System behavior

EvidenceAgent-MM retrieves typed evidence, expands a bounded graph, verifies modality and support requirements, and returns one of three states. The default renderer is deterministic. Qwen3 generation is optional and runs only after evidence selection and gating.

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
