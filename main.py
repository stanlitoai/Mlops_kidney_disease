from Mlops import logger
from Mlops.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Mlops.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Mlops.pipeline.stage_03_training import TrainingPipeline

# from Mlops.pipeline.stage_03_model_training import ModelTrainingPipeline
# sudo apt install nvidia-cuda-toolkit
# from Mlops.pipeline.stage_04_model_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed "
                f"<<<<<<\n\nx==========+++++++++++++++++++++++++++++=========================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Preparing base model"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed "
                f"<<<<<<\n\nx==========+++++++++++++++++++++++++++++=========================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training model"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed "
                f"<<<<<<\n\nx==========+++++++++++++++++++++++++++++=========================x")
except Exception as e:
    logger.exception(e)
    raise e
