import click
from pipeline.training_pipeline import ml_pipeline


@click.command()
def main():
    run = ml_pipeline()
if __name__ == "__main__":
    main()
