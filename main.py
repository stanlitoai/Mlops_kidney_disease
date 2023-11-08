from Mlops import logger
from Mlops.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Mlops.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Mlops.pipeline.stage_03_training import TrainingPipeline
from Mlops.pipeline.stage_04_model_evaluation import EvaluationPipeline

# export MLFLOW_TRACKING_URI=https://dagshub.com/Stanlito-AI/Mlops_kidney_disease.mlflow
#
# export MLFLOW_TRACKING_USERNAME=Stanlito-AI
#
# export MLFLOW_TRACKING_PASSWORD=f616a6150e177ffd52787126e7241eeb312f247f


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


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed "
                f"<<<<<<\n\nx==========+++++++++++++++++++++++++++++=========================x")
except Exception as e:
    logger.exception(e)
    raise e