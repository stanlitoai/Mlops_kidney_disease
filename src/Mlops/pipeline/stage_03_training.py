from Mlops.components.training import Training
from Mlops.config.configuration import ConfigurationManager
from Mlops.components.prepare_base_model import PrepareBaseModel
from Mlops import logger

STAGE_NAME = "Training model"


class TrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===================================x")
    except Exception as e:
        logger.exception(e)
        raise e
