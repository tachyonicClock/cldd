from pathlib import Path

import click
import pandas as pd
from data_profiling import ProfileReport


@click.command()
@click.argument("title")
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("output_html")
def main(title: str, input_csv: str, output_html: str):
    input_csv_ = Path(input_csv)
    output_html_ = Path(output_html)
    output_html_.parent.mkdir(exist_ok=True)

    df = pd.read_csv(input_csv_)
    profile = ProfileReport(df, title=title, explorative=True)
    profile.to_file(output_html_)


if __name__ == "__main__":
    main()
