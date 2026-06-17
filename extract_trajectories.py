import json
from pathlib import Path

import click

METADATA_KEYS = {"usage_metadata", "response_metadata"}


def strip_metadata(conversation_flow):
    cleaned = []
    for entry in conversation_flow:
        cleaned_entry = {k: v for k, v in entry.items() if k not in METADATA_KEYS}
        cleaned.append(cleaned_entry)
    return cleaned


@click.command()
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path), default="results", help="Directory containing result JSON files.")
@click.option("--trajectories-dir", type=click.Path(path_type=Path), default="trajectories", help="Output directory for extracted trajectories.")
def main(results_dir: Path, trajectories_dir: Path):
    for result_file in results_dir.rglob("*.json"):
        relative = result_file.relative_to(results_dir)
        output_file = trajectories_dir / relative

        with open(result_file) as f:
            data = json.load(f)

        runs = data.get("runs", [])
        if not runs:
            continue

        for run in runs:
            conversation_flow = run.get("conversation_flow", [])
            if not conversation_flow:
                continue

            trajectory = {"messages": strip_metadata(conversation_flow)}

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(trajectory, f, indent=2)


if __name__ == "__main__":
    main()
