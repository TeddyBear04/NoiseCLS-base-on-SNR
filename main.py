"""Colab-friendly entrypoint for the 36-label mixture/noise fusion BEATs workflow."""

from __future__ import annotations

import argparse
import sys


COMMANDS = {
    "separator36": "tasks.run_36",
    "head36": "tasks.run_36",
    "finetune36": "tasks.run_36",
    "test36": "tasks.run_36",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", nargs="?")
    arguments, remaining = parser.parse_known_args()
    if arguments.command is None:
        help_parser = argparse.ArgumentParser(description=__doc__)
        help_parser.add_argument("command", choices=COMMANDS)
        help_parser.print_help()
        return
    if arguments.command not in COMMANDS:
        parser.error(f"unknown command: {arguments.command}")
    module = __import__(COMMANDS[arguments.command], fromlist=["main"])
    stage = [arguments.command] if module.__name__ == "tasks.run_36" else []
    sys.argv = [f"main.py {arguments.command}", *stage, *remaining]
    module.main()


if __name__ == "__main__":
    main()
