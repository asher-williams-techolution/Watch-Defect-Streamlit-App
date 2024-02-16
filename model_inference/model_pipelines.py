import os
from typing import Dict, Union, List
from google.cloud import storage
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
import traceback
from pathlib import Path
from .model_wrappers import YOLOModelWrapper
import concurrent.futures
import pandas as pd


class DefectDetectionPipeline():
    """
    A class representing a pipeline for detecting defects in an image using object detection and classification models.

    Attributes:
    - object_detection_model (YOLOModelWrapper): The object detection model for identifying parts of an object.
    - classification_models (dict): A dictionary of classification models for each part, where the key is the part name and the value is the classification model.

    Methods:
    - __init__(self, object_detection_model:YOLOModelWrapper, classification_models:typing.Dict[str, YOLOModelWrapper]): Initializes the DefectDetectionPipeline object.
    - detect_defects(self, input_image:str, confidence_threshold:float = 0.60, iou_threshold:float = 0.60, device:str = None, excluded_parts:typing.List[str] = []): Detects defects in the input image.
    """

    def __init__(self, object_detection_model:YOLOModelWrapper, classification_models:Dict[str, YOLOModelWrapper]):
        """
        Initializes the DefectDetectionPipeline object.

        Parameters:
        - object_detection_model (YOLOModelWrapper): The object detection model for identifying parts of an object.
        - classification_models (dict): A dictionary of classification models for each part, where the key is the part name and the value is the classification model.
        """

        self.object_detection_model = object_detection_model
        self.classification_models = classification_models

        # Validate classification models
        for model in self.classification_models.values():
            if model.model_type != "classification":
                raise ValueError("All models in the classification_models dictionary must be for classification tasks.")

        # Validate part names
        possible_objects = self.object_detection_model.names
        for part_name in self.classification_models.keys():
            if part_name not in possible_objects:
                raise ValueError(f"The part name '{part_name}' is not present in the possible objects the object detection model can find.")

        # Set the local image storage directory for all models to be the same
        image_storage_dir = self.object_detection_model.local_image_storage_dir
        for model in self.classification_models.values():
            model.local_image_storage_dir = image_storage_dir

    # def detect_defects(self, input_image:Union[str, Image.Image], confidence_threshold:float = 0.60, iou_threshold:float = 0.60, device:str = None, excluded_parts:List[str] = []):
    #     """
    #     Detects defects in the input image.

    #     Parameters:
    #     - input_image (str): The file path of the input image.
    #     - confidence_threshold (float, optional): The confidence threshold for object detection. Default is 0.60.
    #     - iou_threshold (float, optional): The intersection over union (IOU) threshold for object detection. Default is 0.60.
    #     - device (str, optional): The device to use for object detection (e.g., "cpu", "cuda"). If not specified, the default device will be used.
    #     - excluded_parts (list, optional): A list of part names to exclude from defect detection.

    #     Returns:
    #     - defect_detection_results (dict): A dictionary containing the defect detection results, including the bounding box, confidence score, cropped image, and classification for each detected defect.
    #     - object_detection_speed (float): The speed of the defect detection process.
    #     - classification_speed (float): The speed of the object classification process.

    #     Raises:
    #     - ValueError: If the model type does not support object detection.
    #     """

    #     if isinstance(input_image, str):
    #         input_image = Image.open(input_image)

    #     # Perform Object Detection
    #     object_detection_results, object_detection_speed = self.object_detection_model.detect_objects(input_image, confidence_threshold, iou_threshold, device)

    #     defect_detection_results = {}

    #     # Perform Object Classification for Each Detected Object
    #     for object_name, object_info in object_detection_results.items():

    #         if object_name in excluded_parts:
    #             defect_detection_results[object_name] = {"Bounding Box": object_info["Bounding Box"], 
    #                                                     "Confidence": object_info["Confidence"], 
    #                                                     "Cropped_Image": object_info["Cropped_Image"],
    #                                                     "Classification": "Excluded",
    #                                                     "Classification_Confidence": None}
    #             continue

    #         cropped_image_path = object_info["Cropped_Image"]
    #         classification_model = self.classification_models[object_name]
    #         classification, confidence, classification_speed = classification_model.classify_object(cropped_image_path, device)
    #         defect_detection_results[object_name] = {"Bounding Box": object_info["Bounding Box"], 
    #                                                  "Confidence": object_info["Confidence"], 
    #                                                  "Cropped_Image": object_info["Cropped_Image"],
    #                                                  "Classification": classification,
    #                                                  "Classification_Confidence": confidence}
        
    #     return defect_detection_results, object_detection_speed, classification_speed


    def detect_defects(self, input_image:Union[str, Image.Image], confidence_threshold:float = 0.60, iou_threshold:float = 0.60, device:str = None, excluded_parts:List[str] = [], return_df:bool = False):
        """
        Detects defects in the input image.

        Parameters:
        - input_image (str): The file path of the input image.
        - confidence_threshold (float, optional): The confidence threshold for object detection. Default is 0.60.
        - iou_threshold (float, optional): The intersection over union (IOU) threshold for object detection. Default is 0.60.
        - device (str, optional): The device to use for object detection (e.g., "cpu", "cuda"). If not specified, the default device will be used.
        - excluded_parts (list, optional): A list of part names to exclude from defect detection.

        Returns:
        - defect_detection_results (dict): A dictionary containing the defect detection results, including the bounding box, confidence score, cropped image, and classification for each detected defect.
        - object_detection_speed (float): The speed of the defect detection process.
        - classification_speed (float): The speed of the object classification process.

        Raises:
        - ValueError: If the model type does not support object detection.
        """

        if isinstance(input_image, str):
            input_image = Image.open(input_image)

        # Perform Object Detection
        object_detection_results, object_detection_speed = self.object_detection_model.detect_objects(input_image, confidence_threshold, iou_threshold, device)

        defect_detection_results = {}

        average_classification_speed = 0

        # Perform Object Classification for Each Detected Object
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for object_name, object_info in object_detection_results.items():
                if object_name in excluded_parts:
                    defect_detection_results[object_name] = {"Bounding Box": object_info["Bounding Box"], 
                                                            "Confidence": round(float(object_info["Confidence"]), 2), 
                                                            "Cropped_Image": object_info["Cropped_Image"],
                                                            "Cropped Image Array": object_info["Cropped Image Array"],
                                                            "Classification": "Non Defective",
                                                            "Classification_Confidence": round(float(object_info["Confidence"]), 2)}
                    continue

                cropped_image_path = object_info["Cropped_Image"]
                classification_model = self.classification_models[object_name]
                future = executor.submit(classification_model.classify_object, cropped_image_path, device)
                futures.append((object_name, future))

            for object_name, future in futures:
                classification, confidence, classification_speed = future.result()

                average_classification_speed += classification_speed["inference"]

                defect_detection_results[object_name] = {"Bounding Box": object_detection_results[object_name]["Bounding Box"], 
                                                        "Confidence": round(float(object_detection_results[object_name]["Confidence"]), 2), 
                                                        "Cropped_Image": object_detection_results[object_name]["Cropped_Image"],
                                                        "Cropped Image Array": object_detection_results[object_name]["Cropped Image Array"],
                                                        "Classification": classification,
                                                        "Classification_Confidence": round(float(confidence), 2)}
            try:
                average_classification_speed /= (len(defect_detection_results) - len(excluded_parts))
            except ZeroDivisionError:
                average_classification_speed = 0

        if return_df:
            defect_detection_results = results_to_df(defect_detection_results)
        else:
            defect_detection_results = defect_detection_results

        return defect_detection_results, object_detection_speed, average_classification_speed
    
def results_to_df(defect_detection_results:Dict[str, Dict[str, Union[str, float]]], columns_to_include:List[str] = ["Classification", "Classification_Confidence"]):
    """
    Converts the defect detection results to a pandas DataFrame.

    Parameters:
    - defect_detection_results (dict): A dictionary containing the defect detection results, including the bounding box, confidence score, cropped image, and classification for each detected defect.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the defect detection results.
    """

    defect_detection_results = {object_name: {key: value for key, value in object_info.items() if key in columns_to_include}
                                for object_name, object_info in defect_detection_results.items()}

    df = pd.DataFrame(defect_detection_results).T
    return df










