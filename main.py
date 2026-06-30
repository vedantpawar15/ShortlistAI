"""Command-line entry point for RecruitAI."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    import typer
except ImportError:
    class _TyperFallback:
        def __init__(self, help: str = "") -> None:
            self.commands: dict[str, object] = {}

        def command(self) -> object:
            def decorator(function: object) -> object:
                self.commands[getattr(function, "__name__")] = function
                return function

            return decorator

        def __call__(self) -> None:
            command_name = sys.argv[1] if len(sys.argv) > 1 else "health"
            command = self.commands.get(command_name)
            if command is None:
                raise SystemExit(f"Unknown command: {command_name}")
            if command_name == "rank":
                output = Path("submission.csv")
                if "--output" in sys.argv:
                    output = Path(sys.argv[sys.argv.index("--output") + 1])
                if "-o" in sys.argv:
                    output = Path(sys.argv[sys.argv.index("-o") + 1])
                command(output)
            else:
                command()

    class _TyperModuleFallback:
        Typer = _TyperFallback

        @staticmethod
        def Option(default: object, *args: object, **kwargs: object) -> object:
            return default

        @staticmethod
        def echo(message: object) -> None:
            print(message)

    typer = _TyperModuleFallback()

from src.config import get_settings
from src.data_loader import DataLoader
from src.logging_utils import logger
from src.ranking_engine import RankingEngine
from src.utils import configure_logging, ensure_directory

cli = typer.Typer(help="RecruitAI offline candidate ranking system.")


@cli.command()
def rank(
    output_path: Path = typer.Option(Path("submission.csv"), "--output", "-o", help="Submission CSV path."),
) -> None:
    """Generate an offline ranked candidate submission from local datasets."""
    settings = get_settings()
    configure_logging(settings.logs_dir)
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates("sample_candidates.json")
    jobs = loader.load_jobs("job_description.docx")
    job = jobs.iloc[0]

    ranked = RankingEngine().rank_candidates(candidates, job)
    try:
        sample_submission = loader.load_submission("sample_submission.csv")
        ranked = align_to_submission_template(ranked, sample_submission)
    except FileNotFoundError:
        logger.info("No sample submission found; writing all ranked candidates")

    ranked["score"] = ranked["score"].round(4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)

    outputs_dir = ensure_directory(settings.outputs_dir)
    ranked.to_csv(outputs_dir / output_path.name, index=False)
    typer.echo(f"Wrote {len(ranked)} ranked candidates to {output_path}")


@cli.command()
def health() -> None:
    """Verify that the CLI can load project configuration."""
    settings = get_settings()
    typer.echo(f"RecruitAI is configured for project: {settings.project_name}")


def align_to_submission_template(ranked: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the template candidate set, ranking scored rows before fallback rows."""
    ranked_frame = ranked.copy()
    sample_frame = sample_submission.copy()
    ranked_ids = set(ranked_frame["candidate_id"].astype(str))
    fallback_rows = []
    for candidate_id in sample_frame["candidate_id"].astype(str):
        if candidate_id in ranked_ids:
            continue
        fallback_rows.append(
            {
                "candidate_id": candidate_id,
                "rank": 0,
                "score": 0.0,
                "reasoning": "Candidate was present in the submission template but absent from sample_candidates.json.",
            }
        )

    if fallback_rows:
        ranked_frame = pd.concat([ranked_frame, pd.DataFrame(fallback_rows)], ignore_index=True)

    ranked_frame = ranked_frame.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    ranked_frame = ranked_frame.head(len(sample_frame))
    ranked_frame["rank"] = range(1, len(ranked_frame) + 1)
    return ranked_frame[["candidate_id", "rank", "score", "reasoning"]]


if __name__ == "__main__":
    cli()

