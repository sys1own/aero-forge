"""Entrypoint adapter engine for generating root-level CLI wrappers."""

import os
from typing import Any, Dict


class EntrypointAdapterEngine:
    """Generates CLI parsing code and entrypoint wrappers for root project execution."""

    def __init__(self, execution_strategy: Dict[str, Any], output_dir: str):
        self.strategy = execution_strategy
        self.output_dir = output_dir

    def synthesize_root_entrypoint(self) -> str:
        primary = self.strategy["primary_entrypoint"]
        cli_contract = self.strategy["cli_contract"]

        if primary["runtime"] == "python3":
            return self._generate_python_cli_wrapper(primary["path"], cli_contract)
        else:
            raise NotImplementedError(
                f"Runtime {primary['runtime']} not supported for auto-wrapper generation."
            )

    def _generate_python_cli_wrapper(
        self, entrypoint_filename: str, cli_contract: Dict[str, Any]
    ) -> str:
        flags = cli_contract.get("flags", [])

        lines = [
            "# Auto-generated entrypoint wrapper produced by Aero-Forge Engine Core",
            "import sys",
            "import argparse",
            "from src.python.engine import run_domain_task",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser(description='Aero-Forge Executable Pipeline')",
        ]

        for flag in flags:
            short_flag = f"-{flag['short']}" if flag.get("short") else ""
            long_flag = f"--{flag['name']}"
            flag_args = [f"'{long_flag}'"]
            if short_flag:
                flag_args.insert(0, f"'{short_flag}'")

            flag_str = ", ".join(flag_args)

            if flag["type"] == "bool":
                lines.append(
                    f"    parser.add_argument({flag_str}, dest='{flag['dest_var']}', "
                    f"action='store_true', help='{flag['help']}')"
                )
            else:
                choices_str = f", choices={flag['choices']}" if flag.get("choices") else ""
                default_str = (
                    f", default={repr(flag['default'])}"
                    if flag.get("default") is not None
                    else ""
                )
                arg_type = (
                    "int"
                    if flag["type"] == "int"
                    else "float"
                    if flag["type"] == "float"
                    else "str"
                )
                lines.append(
                    f"    parser.add_argument({flag_str}, dest='{flag['dest_var']}', "
                    f"type={arg_type}{choices_str}{default_str}, "
                    f"required={flag.get('required', False)}, help='{flag['help']}')"
                )

        lines.extend(
            [
                "    args = parser.parse_args()",
                "    status = run_domain_task(args)",
                "    sys.exit(status if isinstance(status, int) else 0)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )

        code = "\n".join(lines)
        target_path = os.path.join(self.output_dir, entrypoint_filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)
        return target_path
