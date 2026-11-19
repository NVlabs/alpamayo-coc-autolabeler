"""Generate streaming RSG/CoC labels and export completed public labels."""

import argparse
import json
from pathlib import Path

from .configuration import SEARCH_KNOBS, GenerationSetupError, load_config, validate_setup


def main() -> None:
    """Run label generation, resume an interrupted run, or export completed labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("run", help="generate streaming RSG/CoC labels")
    generate.add_argument(
        "--config", type=Path, required=True, help="local generation YAML or JSON"
    )
    generate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="persistent run directory containing journals and publication",
    )
    generate.add_argument(
        "--resume",
        action="store_true",
        help="reuse this run with unchanged inputs, code and model settings",
    )
    generate.add_argument(
        "--recovery-source", type=Path, help="prior run to recover into a new output directory"
    )
    generate.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable experimental override; allowed keys: " + ", ".join(sorted(SEARCH_KNOBS)),
    )
    prepare = commands.add_parser(
        "prepare-video", help="bind front-wide MP4 inputs without API calls"
    )
    prepare.add_argument(
        "--config",
        type=Path,
        required=True,
        help="native generation template with compiler and registry paths",
    )
    prepare.add_argument(
        "--videos", type=Path, required=True, help="directory containing front-facing MP4 inputs"
    )
    prepare.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new directory for prepared manifests and generation config",
    )
    export = commands.add_parser("export-public", help="export completed labels")
    export.add_argument(
        "--publication", type=Path, required=True, help="compact clips publication directory"
    )
    export.add_argument("--clip", required=True, help="clip ID present in that publication")
    export.add_argument(
        "--out", type=Path, required=True, help="new export directory; must not already exist"
    )
    annotation = commands.add_parser("export-annotation", help="export graph and captions as JSON")
    annotation.add_argument(
        "--publication", type=Path, required=True, help="compact clips publication directory"
    )
    annotation.add_argument("--clip", required=True, help="clip ID present in that publication")
    annotation.add_argument(
        "--out", type=Path, required=True, help="new export directory; must not already exist"
    )
    annotation.add_argument(
        "--diagnostics", action="store_true", help="include a separate internal audit record"
    )
    args = parser.parse_args()
    if args.command in {"run", "prepare-video"} and not args.config.is_file():
        parser.error(f"generation config file does not exist: {args.config}")
    if args.command == "run":
        from .interface import run

        try:
            summary = run(
                args.config,
                out_dir=args.out,
                overrides=args.override,
                resume=args.resume,
                recovery_source=args.recovery_source,
            )
        except GenerationSetupError as error:
            parser.error(str(error))
        print(json.dumps(summary))
        if not summary["success"]:
            raise SystemExit(1)
    elif args.command == "prepare-video":
        import yaml

        from .video_inputs import prepare_video_inputs

        try:
            generation = load_config(args.config)
            validate_setup(generation, preparation=True)
        except GenerationSetupError as error:
            parser.error(str(error))
        if generation["mode"] != "native":
            raise ValueError("prepare-video requires native generation mode")
        compiler = yaml.safe_load(Path(generation["compiler_config"]).read_text())
        compiler["infra"]["endpoint_registry"] = generation["endpoint_registry"]
        print(json.dumps(prepare_video_inputs(args.videos, args.out, compiler, generation)))
    elif args.command == "export-annotation":
        from .export.publication import write_annotation
        from .export.storage import load_clip_records

        write_annotation(
            load_clip_records(args.publication)[args.clip], args.out, diagnostics=args.diagnostics
        )
    else:
        from .export.publication import write_public_store
        from .export.storage import load_clip_records

        write_public_store(load_clip_records(args.publication)[args.clip]["result"], args.out)


if __name__ == "__main__":
    main()
