from mystic_forge.pipeline import Pipeline
from mystic_forge.stages.fetch_cards import FetchCardsStage
from mystic_forge.stages.preprocess import PreprocessStage


def main() -> None:
    Pipeline([
        FetchCardsStage(),
        PreprocessStage(),
    ]).run()


if __name__ == "__main__":
    main()
