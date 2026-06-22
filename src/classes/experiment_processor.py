import os
from loguru import logger

from .data_processor import DataProcessor

class ExperimentProcessor:
    @staticmethod
    def process_experiments(
        base_path,
        group=None,
        experiment=None,
        subject=None,
        output_path=None,
        skip_existing=False,
    ):

        for group_name in os.listdir(base_path):
            if group and group_name != group:
                continue

            group_path = os.path.join(base_path, group_name)
            if not os.path.isdir(group_path):
                logger.info(f"Skipping non-directory in raw root: {group_path}")
                continue

            for experiment_name in os.listdir(group_path):
                if experiment and experiment_name != experiment:
                    continue

                experiment_path = os.path.join(group_path, experiment_name)
                if not os.path.isdir(experiment_path):
                    logger.info(f"Skipping non-directory in group folder: {experiment_path}")
                    continue

                for subject_name in os.listdir(experiment_path):
                    if subject and subject_name != subject:
                        continue

                    subject_path = os.path.join(experiment_path, subject_name)
                    if not os.path.isdir(subject_path):
                        logger.info(f"Skipping non-directory in experiment folder: {subject_path}")
                        continue

                    output_dir = os.path.join(
                        output_path,
                        group_name,
                        experiment_name
                    )

                    filename = os.path.join(
                        output_dir,
                        f"{subject_name}.csv"
                    )

                    if skip_existing and os.path.exists(filename):
                        logger.info(f"[SKIP] Existing output: {filename}")
                        continue

                    logger.info(f"Processing {group_name} | {experiment_name} | {subject_name}")
                    data = []

                    for element in ["images", "labels", "watch_sensors"]:
                        subject_element_path = os.path.join(subject_path, element)

                        if "__" in subject_element_path:
                            continue

                        data.extend(
                            DataProcessor.process_subject(subject_element_path, group_name)
                        )

                    os.makedirs(output_dir, exist_ok=True)

                    DataProcessor.save_data_df(data, filename)
