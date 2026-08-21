"""CLI for the offline scene-domain enrichment job."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import config
from .enricher import DomainEnricher
from .llm import LiteLLMDomainEnricher
from .repository import DomainRepository
from .service import ProcessStatus, create_s3_client, discover_sources, process_source


class CliArgs(argparse.Namespace):
    prefix: str = ""
    groups: list[str] = []
    video: str = ""
    file_list: str = ""
    workers: int = 4
    model: str = ""
    semantic_retries: int = 3
    force: bool = False
    dry_run: bool = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("phải >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("phải >= 0")
    return parsed


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Tag temporal scene domains with an LLM and persist to MongoDB."
    )
    _ = parser.add_argument("--prefix", default="")
    _ = parser.add_argument(
        "--groups",
        nargs="+",
        default=[],
        metavar="GROUP",
        help="Nhiều group, nhận tên trần hoặc đầy đủ: --groups L26_b L26_c "
        "(hoặc Keyscence_L26_b). Chia việc theo người thì mỗi người một danh sách "
        "rời nhau.",
    )
    _ = parser.add_argument("--video", default="")
    _ = parser.add_argument("--file-list", default="")
    _ = parser.add_argument("--workers", type=positive_int, default=4)
    _ = parser.add_argument(
        "--model",
        default="",
        help=f'litellm "provider/model", mặc định {config.domain_llm_model!r}. '
        "Ví dụ: cerebras/gpt-oss-120b, gemini/gemini-3.7-flash",
    )
    _ = parser.add_argument("--semantic-retries", type=non_negative_int, default=3)
    _ = parser.add_argument("--force", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(namespace=CliArgs())


def create_enricher(args: CliArgs) -> DomainEnricher:
    return LiteLLMDomainEnricher(
        model=args.model or None, semantic_retries=args.semantic_retries
    )


def main() -> int:
    args = parse_args()
    s3 = create_s3_client()
    sources = discover_sources(
        s3,
        config.aws_bucket_name,
        prefix=args.prefix,
        groups=args.groups,
        video_id=args.video,
        file_list=args.file_list,
    )
    scope = (
        f"groups={args.groups}" if args.groups
        else f"prefix={args.prefix!r}" if args.prefix
        else "toàn bộ bucket"
    )
    print(f"model={config.domain_llm_model if not args.model else args.model} {scope}")
    print(f"found={len(sources)} scenes.json")
    if args.dry_run:
        for source in sources:
            print(source.source_id)
        return 0
    if not sources:
        return 0

    enricher = create_enricher(args)
    repository = DomainRepository()
    counts: Counter[ProcessStatus] = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_source,
                    source,
                    s3,
                    enricher,
                    repository,
                    force=args.force,
                ): source
                for source in sources
            }
            for future in as_completed(futures):
                result = future.result()
                counts[result.status] += 1
                if result.status in {ProcessStatus.ERROR, ProcessStatus.INVALID}:
                    print(f"[{result.video_id}] {result.status}: {result.message}")
    finally:
        repository.close()
        enricher.close()

    print({status.value: count for status, count in counts.items()})
    failures = counts[ProcessStatus.ERROR] + counts[ProcessStatus.INVALID]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
