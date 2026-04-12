from mystic_forge.logger import logger
from mystic_forge.pipeline import Pipeline
from mystic_forge.stages.fetch import FetchStage
from mystic_forge.stages.preprocess import PreprocessStage
from mystic_forge.stages.train import TrainStage
from mystic_forge.stages.sample import SampleStage
from mystic_forge.stages.validate import ValidateStage


def main() -> None:
    try:
        Pipeline([
            FetchStage(),
            PreprocessStage(),
            TrainStage(),
            SampleStage(),
            ValidateStage(),
        ]).run()
    except Exception:
        logger.error("Pipeline aborted")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
