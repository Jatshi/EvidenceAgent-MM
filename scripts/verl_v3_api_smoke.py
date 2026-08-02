"""Exercise EvidenceAgent tools through verl's real v0.8 tool registry."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path


async def run_smoke(tool_config: Path, evidence_store: Path) -> dict[str, object]:
    from verl.tools.tool_registry import load_all_tools

    tools = load_all_tools(tool_config_path=str(tool_config), function_tool_path=None)
    by_name = {tool.name: tool for tool in tools}
    if set(by_name) != {"evidence_search", "verify_claim_support"}:
        raise RuntimeError(f"unexpected tool registry: {sorted(by_name)}")

    search = by_name["evidence_search"]
    search_id, _ = await search.create(
        session_id="v3-smoke",
        evidence_store=str(evidence_store),
        max_steps=4,
        max_unique_evidence=8,
        max_tool_time_ms=5_000,
    )
    response, search_reward, search_metrics = await search.execute(
        search_id,
        {"query": "Who proposed design B and what was the latency?", "top_k": 2},
    )
    search_payload = json.loads(response.text)
    await search.release(search_id)

    verify = by_name["verify_claim_support"]
    verify_id, _ = await verify.create(
        session_id="v3-smoke",
        evidence_store=str(evidence_store),
    )
    verify_response, verify_reward, verify_metrics = await verify.execute(
        verify_id,
        {
            "claim": "SPEAKER_00 proposed design B with 42 ms latency.",
            "evidence_ids": "smoke:utt:1,smoke:ocr:7",
        },
    )
    verify_payload = json.loads(verify_response.text)
    await verify.release(verify_id)
    if not search_payload["results"] or not verify_payload["verified"]:
        raise RuntimeError("real verl tool registry smoke did not produce verified evidence")
    return {
        "status": "passed",
        "verl_version": importlib.metadata.version("verl"),
        "tools": sorted(by_name),
        "search_reward": search_reward,
        "search_metrics": search_metrics,
        "verify_reward": verify_reward,
        "verify_metrics": verify_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-config", type=Path, default=Path("configs/verl/tools_v3.json"))
    parser.add_argument(
        "--evidence-store",
        type=Path,
        default=Path("benchmarks/eamm_v3_hard/smoke_store.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3/smoke/verl_tools.json"))
    args = parser.parse_args()
    report = asyncio.run(run_smoke(args.tool_config.resolve(), args.evidence_store.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
