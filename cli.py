import os
import sys
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from analyzer import TwitterSentimentAnalyzer
from preprocessor import TweetPreprocessor

console = Console()

COLOR_MAP = {
    "positive": "bold green",
    "negative": "bold red",
    "neutral": "bold cyan",
}

EMOJI_MAP = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}


def display_single_result(result: dict, show_spacy: bool = False, preprocessor: TweetPreprocessor = None):
    sentiment = result["sentiment"]
    color = COLOR_MAP.get(sentiment, "white")
    emoji_icon = EMOJI_MAP.get(sentiment, "")
    confidence = result.get("confidence", 0.0)

    title = f"[{color}]Sentiment: {sentiment.upper()} {emoji_icon} (Confidence: {confidence * 100:.1f}%)[/{color}]"

    table = Table(box=box.ROUNDED, show_header=False, expand=True)
    table.add_column("Field", style="bold yellow", width=18)
    table.add_column("Value")

    table.add_row("Raw Tweet", result["raw_text"])
    table.add_row("Method Used", result["method"].upper())

    if "compound_score" in result:
        comp = result["compound_score"]
        comp_color = "green" if comp >= 0.05 else ("red" if comp <= -0.05 else "cyan")
        table.add_row("Compound Score", f"[{comp_color}]{comp:+.4f}[/{comp_color}]")

    if "breakdown" in result:
        bd = result["breakdown"]
        bd_str = f"[green]Pos: {bd['positive']:.3f}[/green] | [cyan]Neu: {bd['neutral']:.3f}[/cyan] | [red]Neg: {bd['negative']:.3f}[/red]"
        table.add_row("VADER Breakdown", bd_str)

    if "probabilities" in result:
        probs = result["probabilities"]
        prob_str = " | ".join([f"{k.capitalize()}: {v * 100:.1f}%" for k, v in probs.items()])
        table.add_row("ML Probabilities", prob_str)

    meta = result.get("metadata", {})
    if meta.get("hashtags"):
        table.add_row("Hashtags", ", ".join([f"#{h}" for h in meta["hashtags"]]))
    if meta.get("mentions"):
        table.add_row("Mentions", ", ".join(meta["mentions"]))
    if meta.get("emojis"):
        table.add_row("Emojis", " ".join(meta["emojis"]))

    console.print(Panel(table, title=title, border_style=color.split()[-1]))

    if show_spacy and preprocessor:
        features = preprocessor.spacy_linguistic_features(result["raw_text"])
        if features:
            spacy_table = Table(title="SpaCy Linguistic Analysis", box=box.SIMPLE_HEAVY)
            spacy_table.add_column("Token", style="bold")
            spacy_table.add_column("Lemma", style="magenta")
            spacy_table.add_column("POS", style="cyan")
            spacy_table.add_column("Dep", style="yellow")
            spacy_table.add_column("Stop Word?", style="dim")

            for feat in features[:15]:
                spacy_table.add_row(
                    feat["text"], feat["lemma"], feat["pos"], feat["dep"], feat["is_stop"]
                )
            console.print(spacy_table)


def run_batch_analysis(
    analyzer: TwitterSentimentAnalyzer,
    input_file: str,
    output_file: str = None,
    text_col: str = "tweet",
    method: str = "vader",
):
    if not os.path.exists(input_file):
        console.print(f"[bold red]Error:[/bold red] File not found: {input_file}")
        return

    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith((".json", ".jsonl")):
        df = pd.read_json(input_file)
    elif input_file.endswith(".txt"):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({text_col: lines})
    else:
        console.print("[bold red]Unsupported file format. Please provide .csv, .json, or .txt[/bold red]")
        return

    enriched_df = analyzer.analyze_batch(df, text_column=text_col, method=method)
    metrics = analyzer.get_summary_metrics(enriched_df)

    summary_table = Table(title="Batch Analysis Summary Metrics", box=box.ROUNDED)
    summary_table.add_column("Metric", style="bold yellow")
    summary_table.add_column("Value", style="bold")

    summary_table.add_row("Total Tweets Analyzed", str(metrics["total_tweets"]))
    for sent, count in metrics["sentiment_counts"].items():
        pct = metrics["sentiment_percentages"][sent]
        color = COLOR_MAP.get(sent, "white")
        summary_table.add_row(
            f"{sent.capitalize()} Tweets",
            f"[{color}]{count} ({pct:.1f}%)[/{color}]",
        )
    summary_table.add_row("Average Confidence", f"{metrics['average_confidence'] * 100:.1f}%")
    summary_table.add_row("Average Compound Score", f"{metrics['average_compound']:+.4f}")

    console.print(summary_table)

    preview_table = Table(title="Sample Results Preview (First 5 Tweets)", box=box.SIMPLE_HEAD)
    preview_table.add_column("Tweet", style="dim", max_width=45, overflow="ellipsis")
    preview_table.add_column("Sentiment", style="bold")
    preview_table.add_column("Confidence")
    preview_table.add_column("Score")

    for _, row in enriched_df.head(5).iterrows():
        color = COLOR_MAP.get(row["sentiment"], "white")
        preview_table.add_row(
            str(row[text_col]),
            f"[{color}]{row['sentiment'].upper()}[/{color}]",
            f"{row['confidence'] * 100:.1f}%",
            f"{row['compound_score']:+.3f}",
        )
    console.print(preview_table)

    if output_file:
        enriched_df.to_csv(output_file, index=False)
        console.print(f"[bold green]Saved to:[/bold green] {output_file}")


def run_interactive_mode(analyzer: TwitterSentimentAnalyzer, method: str = "vader", show_spacy: bool = False):
    console.print(Panel(
        "[bold cyan]Twitter Sentiment Shell[/bold cyan]\n"
        "Enter tweet text to analyze. Type [bold red]'exit'[/bold red] to quit.\n"
        "Switch model using [bold magenta]'switch vader|ml|ensemble'[/bold magenta].",
        box=box.ROUNDED,
    ))

    current_method = method
    while True:
        try:
            user_input = console.input(f"\n[bold yellow]Tweet ({current_method.upper()}) > [/bold yellow]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break
            if user_input.lower().startswith("switch "):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].lower() in ("vader", "ml", "ensemble"):
                    current_method = parts[1].lower()
                    console.print(f"[bold green]Engine: {current_method.upper()}[/bold green]")
                    continue

            result = analyzer.analyze(user_input, method=current_method)
            display_single_result(result, show_spacy=show_spacy, preprocessor=analyzer.preprocessor)
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis")
    parser.add_argument("--text", "-t", type=str, help="Single tweet text to analyze.")
    parser.add_argument("--file", "-f", type=str, help="Path to CSV, JSON, or TXT file.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path.")
    parser.add_argument("--col", type=str, default="tweet", help="Column name.")
    parser.add_argument(
        "--method", "-m",
        choices=["vader", "ml", "ensemble"],
        default="vader",
        help="Sentiment analysis method.",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode.")
    parser.add_argument("--spacy", action="store_true", help="Show SpaCy features.")

    args = parser.parse_args()
    analyzer = TwitterSentimentAnalyzer()

    if args.text:
        result = analyzer.analyze(args.text, method=args.method)
        display_single_result(result, show_spacy=args.spacy, preprocessor=analyzer.preprocessor)
    elif args.file:
        run_batch_analysis(
            analyzer=analyzer,
            input_file=args.file,
            output_file=args.output,
            text_col=args.col,
            method=args.method,
        )
    elif args.interactive or len(sys.argv) == 1:
        run_interactive_mode(analyzer, method=args.method, show_spacy=args.spacy)


if __name__ == "__main__":
    main()
