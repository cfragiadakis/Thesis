import os
from classes.data_processor import DataProcessor
from loguru import logger

class ExperimentProcessor:
    @staticmethod
    def process_experiments(base_path, group=None, experiment=None, subject=None, output_path=None):

        for group_name in os.listdir(base_path):
            if group and group_name != group:
                continue

            group_path = os.path.join(base_path, group_name)

            for experiment_name in os.listdir(group_path):
                if experiment and experiment_name != experiment:
                    continue

                experiment_path = os.path.join(group_path, experiment_name)

                if os.path.isdir(experiment_path):

                    for subject_name in os.listdir(experiment_path):
                        if subject and subject_name != subject:
                            continue

                        logger.info(f"Processing {group_name} | {experiment_name} | {subject_name}")

                        subject_path = os.path.join(experiment_path, subject_name)

                        if os.path.isdir(subject_path):
                            data = []

                            for element in ["images", "labels", "watch_sensors"]:
                                subject_element_path = os.path.join(subject_path, element)

                                if "__" in subject_element_path:
                                    continue

                                data.extend(
                                    DataProcessor.process_subject(subject_element_path, group_name)
                                )

                            output_dir = os.path.join(
                                output_path,
                                group_name,
                                experiment_name
                            )

                            os.makedirs(output_dir, exist_ok=True)

                            filename = os.path.join(
                                output_dir,
                                f"{subject_name}.csv"
                            )

                            DataProcessor.save_data_df(data, filename)