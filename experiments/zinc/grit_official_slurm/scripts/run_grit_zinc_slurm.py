#!/usr/bin/env python3
"""Run official GRIT ZINC jobs with a parameter-count guard."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


CFG_BY_VARIANT = {
    "official": "configs/GRIT/zinc-GRIT-RRWP.yaml",
    "1hop": "configs/GRIT/zinc-GRIT-RRWP-1hop.yaml",
}
NAME_TAG_BY_VARIANT = {
    "official": "slurm.official.GRITwRRWP",
    "1hop": "slurm.1hop.GRITwRRWP",
}
EXPECTED_PARAMS = 473_473
PARAM_RE = re.compile(r"Num parameters:\s*([0-9,]+)")


def build_command(args: argparse.Namespace) -> list[str]:
    cfg = CFG_BY_VARIANT[args.variant]
    name_tag = args.name_tag or NAME_TAG_BY_VARIANT[args.variant]
    cmd = [
        args.python,
        "-u",
        "main.py",
        "--cfg",
        cfg,
        "wandb.use",
        "False",
        "accelerator",
        args.accelerator,
        "dataset.dir",
        str(args.data_dir),
        "out_dir",
        str(args.out_dir),
        "seed",
        str(args.seed),
        "name_tag",
        name_tag,
    ]
    if args.max_epoch is not None:
        cmd.extend(["optim.max_epoch", str(args.max_epoch)])
    cmd.extend(args.cfg_overrides)
    return cmd


def run_and_guard(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / "main.py").is_file():
        raise FileNotFoundError(f"GRIT repo does not contain main.py: {repo_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{args.variant}_seed{args.seed}_{stamp}.log"

    cmd = build_command(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("[run] " + " ".join(map(str, cmd)), flush=True)
    print(f"[run] cwd={repo_dir}", flush=True)
    print(f"[run] log={log_file}", flush=True)

    saw_params = False
    with log_file.open("w", encoding="utf-8") as log:
        log.write("[run] " + " ".join(map(str, cmd)) + "\n")
        log.write(f"[run] cwd={repo_dir}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            match = PARAM_RE.search(line)
            if match and not saw_params:
                saw_params = True
                actual = int(match.group(1).replace(",", ""))
                msg = (
                    f"[param-check] observed={actual} "
                    f"expected={args.expected_params}\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                log.write(msg)
                log.flush()
                if actual != args.expected_params and not args.allow_param_count_drift:
                    err = (
                        "[param-check:ERROR] Parameter count mismatch; "
                        "terminating before training continues.\n"
                    )
                    sys.stdout.write(err)
                    sys.stdout.flush()
                    log.write(err)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    return 2
        rc = proc.wait()

    if not saw_params:
        print("[param-check:ERROR] Did not observe `Num parameters:` in GRIT output.")
        return 3 if rc == 0 else rc
    return rc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official GRIT or the parameter-matched 1-hop GRIT control on ZINC."
    )
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(CFG_BY_VARIANT), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--name-tag", default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--expected-params", type=int, default=EXPECTED_PARAMS)
    parser.add_argument("--allow-param-count-drift", action="store_true")
    parser.add_argument("cfg_overrides", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_and_guard(args)


if __name__ == "__main__":
    raise SystemExit(main())
