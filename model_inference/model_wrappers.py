import os
from google.cloud import storage
import cv2
import PIL
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
import traceback
from pathlib import Path
import shutil
from typing import Union
from PIL import Image

# def initialize_models(GCS_bucket_name:str, GCS_model_directory:str, GCS_Auth_JSON_Path:str, model_file_names:list, local_model_storage_dir:str = None, local_image_storage_dir:str=None, model_type:str = None):
#     """
#     Initializes multiple model wrappers for the models located in the same GCS directory.

#     Parameters:
#     - GCS_bucket_name (str): The name of the Google Cloud Storage (GCS) bucket.
#     - GCS_model_directory (str): The directory path in the GCS bucket where the models are stored.
#     - GCS_Auth_JSON_Path (str): The file path to the GCS authentication JSON file.
#     - model_file_names (list): A list of model file names in the GCS bucket.
#     - local_model_storage_dir (str, optional): The directory path for storing the downloaded models locally. If not provided, a default directory will be created.
#     - local_image_storage_dir (str, optional): The directory path for storing the input images locally. If not provided, a default directory will be created.
#     - model_type (str, optional): The type of the models (classification, detection, or segmentation). If not provided, the model type will be determined based on the model task.

#     Returns:
#     - model_wrappers (list): A list of initialized model wrappers.
#     """
#     model_wrappers = []
#     for model_file_name in model_file_names:
#         model_wrapper = YOLOModelWrapper(GCS_bucket_name, GCS_model_directory, model_file_name, GCS_Auth_JSON_Path, local_model_storage_dir, local_image_storage_dir, model_type)
#         model_wrappers.append(model_wrapper)
#     return model_wrappers


class YOLOModelWrapper():

    """
    Purpose:
    This class downloads and wraps a YOLO model for the purpose of object detection or classification
    for defect detection use cases.
    
    Attributes:
    - allowed_model_types (list): A list of allowed model types.
    - model_task_to_type_map (dict): A mapping of model tasks to model types.
    - GCS_bucket_name (str): The name of the Google Cloud Storage (GCS) bucket.
    - GCS_model_directory (str): The directory path in the GCS bucket where the model is stored.
    - GCS_model_file_name (str): The name of the model file in the GCS bucket.
    - GCS_Auth_JSON_Path (str): The file path to the GCS authentication JSON file.
    - local_model_storage_dir (str): The directory path for storing the downloaded model locally.
    - local_image_storage_dir (str): The directory path for storing the input images locally.
    - model_type (str): The type of the model (classification, detection, or segmentation).

    Methods:
    - __init__(self, GCS_bucket_name, GCS_model_directory, GCS_model_file_name, GCS_Auth_JSON_Path, local_model_storage_dir=None, local_image_storage_dir=None, model_type=None): Initializes the YOLOModelWrapper object.
    - initialize_gcs_bucket(self, bucket_name): Initializes a connection to a Google Cloud Storage (GCS) bucket.
    - get_model_type(self, model_type): Returns the model type based on the provided model task or the default model type if not specified.
    - download_model(self): Downloads the model from the GCS bucket to the local storage.
    - initialize_model(self): Initializes the YOLO model using the downloaded model file.
    - is_model_downloaded(self): Checks if the model is already downloaded.
    - get_model(self): Downloads and initializes the model if it is not already downloaded.
    - reset_image_directory(self): Deletes all files in the image directory and creates a new empty directory.
    - detect_objects(self, input_image_path, confidence_threshold=0.60, iou_threshold=0.60, device=None): Performs object detection on the input image.
    - classify_object(self, input_image_path, device=None): Performs object classification on the input image.
    """

    allowed_model_types = ["classification", "detection", "segmentation"]
    model_task_to_type_map = {"classify": "classification", "detect": "detection", "segment": "segmentation"}

    def __init__(self, GCS_bucket_name:str, GCS_model_directory:str, GCS_model_file_name:str,
                 GCS_Auth_JSON_Path:str=None, local_model_storage_dir:str = None, local_image_storage_dir:str=None, model_type:str = None):
        """
        Initializes the YOLOModelWrapper object.

        Parameters:
        - GCS_bucket_name (str): The name of the Google Cloud Storage (GCS) bucket.
        - GCS_model_directory (str): The directory path in the GCS bucket where the model is stored.
        - GCS_model_file_name (str): The name of the model file in the GCS bucket.
        - GCS_Auth_JSON_Path (str): The file path to the GCS authentication JSON file.
        - local_model_storage_dir (str, optional): The directory path for storing the downloaded model locally. If not provided, a default directory will be created.
        - local_image_storage_dir (str, optional): The directory path for storing the input images locally. If not provided, a default directory will be created.
        - model_type (str, optional): The type of the model (classification, detection, or segmentation). If not provided, the model type will be determined based on the model task.
        """
        
        # Initialize GCS Storage Bucket
        self.GCS_bucket_name = GCS_bucket_name
        self.GCS_model_directory = GCS_model_directory
        self.GCS_model_file_name = GCS_model_file_name
        self.GCS_auth_JSON_path = GCS_Auth_JSON_Path

        if self.GCS_auth_JSON_path is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GCS_auth_JSON_path

        self.GCS_storage_bucket = self.initialize_gcs_bucket(self.GCS_bucket_name)

        # Initialize Local Storage Directories
        self.local_model_storage_dir = self._create_local_model_storage_dir(local_model_storage_dir)
        self.local_image_storage_dir = self._create_local_image_storage_dir(local_image_storage_dir)
        self.local_model_path = os.path.join(self.local_model_storage_dir, self.GCS_model_file_name)
        
        # Initialize Model
        if self.is_model_downloaded():
            print("Model already downloaded")
            self.model = self.initialize_model()
        else:
            print("Model not downloaded, fetching now from GCS")
            self.model = self.get_model()

        # Initialize all possible object names
        self.names = list(self.model.names.values())
        
        # Initialize Model Type
        self.model_type = self.get_model_type(model_type)

    def _create_local_image_storage_dir(self, local_image_storage_dir:str):
        """
        Creates a local image storage directory.

        Parameters:
        - local_image_storage_dir (str): The directory path for storing the input images locally.

        Returns:
        - local_image_storage_dir (str): The directory path for storing the input images locally.
        """

        if local_image_storage_dir is None:
            local_image_storage_dir = os.path.join(os.getcwd(), "images")

        os.makedirs(local_image_storage_dir, exist_ok=True)
        return local_image_storage_dir
    
    def _create_local_model_storage_dir(self, local_model_storage_dir:str):
        """
        Creates a local model storage directory.

        Parameters:
        - local_model_storage_dir (str): The directory path for storing the downloaded model locally.

        Returns:
        - local_model_storage_dir (str): The directory path for storing the downloaded model locally.
        """

        if local_model_storage_dir is None:
            local_model_storage_dir = os.path.join(os.getcwd(), "models")

        os.makedirs(local_model_storage_dir, exist_ok=True)
        return local_model_storage_dir

    def initialize_gcs_bucket(self, bucket_name):
        """
        Initializes a connection to a Google Cloud Storage (GCS) bucket.

        Parameters:
        - bucket_name (str): The name of the GCS bucket.

        Returns:
        - bucket (google.cloud.storage.bucket.Bucket): The GCS bucket object.

        Raises:
        - Exception: If there is an error connecting to the GCS bucket.
        """
        
        try:
            bucket = storage.Client().bucket(bucket_name)
            print("Successfully connected to GCS Storage Bucket")
            return bucket
        except Exception as e:
            print(f"Error connecting to GCS Storage Bucket: {e}")
            print(traceback.format_exc())
            

    def get_model_type(self, model_type:str = None):
        """
        Returns the model type based on the provided model task or the default model type if not specified.

        Parameters:
        - model_type (str, optional): The model task or model type.

        Returns:
        - model_type (str): The model type.

        Raises:
        - ValueError: If the provided model type is invalid.
        """
        if model_type is None:
            return self.model_task_to_type_map[self.model.task]
        else:
            if model_type in self.allowed_model_types:
                return model_type
            else:
                if model_type in self.model_task_to_type_map.keys():
                    return self.model_task_to_type_map[model_type]
                else:
                    raise ValueError(f"Invalid model type: {model_type}. Please choose from {self.allowed_model_types} or {list(self.model_task_to_type_map.keys())}")
                    return None

    def download_model(self):
        """
        Downloads the model from the GCS bucket to the local storage.

        Raises:
        - Exception: If there is an error downloading the model from GCS.
        """

        try:
            gcs_model_path = os.path.join(self.GCS_model_directory, self.GCS_model_file_name).replace("\\", "/")
            self.GCS_storage_bucket.blob(gcs_model_path).download_to_filename(self.local_model_path)
            print(f"Model downloaded to {self.local_model_path}")
        except Exception as e:
            print(f"Error downloading model from GCS: {e}")
            print(traceback.format_exc())
    
    def initialize_model(self):
        """
        Initializes the YOLO model using the downloaded model file.

        Returns:
        - model (YOLO): The initialized YOLO model.

        Raises:
        - Exception: If there is an error initializing the model.
        """

        try:
            model = YOLO(self.local_model_path)
            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            print(traceback.format_exc())
            quit()

    def is_model_downloaded(self):
        """
        Checks if the model is already downloaded.

        Returns:
        - is_downloaded (bool): True if the model is downloaded, False otherwise.
        """

        return os.path.exists(self.local_model_path)

    def get_model(self):
        """
        Downloads and initializes the model if it is not already downloaded.

        Returns:
        - model (YOLO): The downloaded and initialized YOLO model.
        """

        self.download_model()
        model = self.initialize_model()
        return model
    
    def reset_image_directory(self):
        """
        Deletes all files in the image directory and creates a new empty directory.
        """

        shutil.rmtree(self.local_image_storage_dir)
        os.makedirs(self.local_image_storage_dir, exist_ok=True)
    
    def detect_objects(self, input_image: Union[str, Image.Image], confidence_threshold: float = 0.60, iou_threshold: float = 0.60, device: str = None):
        """
        Performs object detection on the input image.

        Parameters:
        - input_image (Union[str, Image.Image]): The file path or PIL image object of the input image.
        - confidence_threshold (float, optional): The confidence threshold for object detection. Default is 0.60.
        - iou_threshold (float, optional): The intersection over union (IOU) threshold for object detection. Default is 0.60.
        - device (str, optional): The device to use for object detection (e.g., "cpu", "cuda"). If not specified, the default device will be used.

        Returns:
        - object_detection_results (dict): A dictionary containing the object detection results, including the bounding box, confidence score, and cropped image for each detected object.
        - speed (float): The speed of the object detection process.

        Raises:
        - ValueError: If the model type does not support object detection.
        """

        if self.model_type not in ["detection", "segmentation"]:
            raise ValueError(f"Model type {self.model_type} does not support object detection")
            return None
        
        self.reset_image_directory()

        # Convert input_image to PIL image object if it's a file path
        if isinstance(input_image, str):
            input_image = Image.open(input_image)

        # Perform Object Detection
        detection_results = self.model.predict(input_image, device=device, conf=confidence_threshold, iou=iou_threshold)

        # This only takes one image at a time so we can just take the first element
        detection_results = detection_results[0]
        original_image = detection_results.orig_img
        detected_object_names = detection_results.names
        object_bounding_boxes = detection_results.boxes.cpu()
        speed = detection_results.speed
        
        object_detection_results = {}
        
        # Loop through each object detected
        for box in object_bounding_boxes:
            object_name = detected_object_names[int(box.cls)]
            confidence_score = box.conf
            bounding_box = box.xyxy[0]

            path_to_cropped_image = Path(os.path.join(self.local_image_storage_dir, f"{object_name}_cropped.jpg"))
            cropped_image = save_one_box(bounding_box, original_image, path_to_cropped_image)

            object_detection_results[object_name] = {"Bounding Box": bounding_box, 
                                                     "Confidence": confidence_score, 
                                                     "Cropped_Image": path_to_cropped_image,
                                                     "Cropped Image Array": cropped_image}
            
        return object_detection_results, speed
    
    def classify_object(self, input_image:Union[str, Image.Image], device: str = None):
        """
        Performs object classification on the input image.

        Parameters:
        - input_image_path (str): The file path of the input image.
        - device (str, optional): The device to use for object classification (e.g., "cpu", "cuda"). If not specified, the default device will be used.

        Returns:
        - classification (str): The predicted class label.
        - confidence (float): The confidence score of the predicted class.
        - speed (float): The speed of the object classification process.

        Raises:
        - ValueError: If the model type does not support object classification.
        """

        if self.model_type not in ["classification"]:
            raise ValueError(f"Model type {self.model_type} does not support object classification")
            return None
        
        # Convert input_image to PIL image object if it's a file path
        if isinstance(input_image, str):
            input_image = Image.open(input_image)
        
        # Perform Object Classification
        classification_results = self.model.predict(input_image, device=device)

        # Only One Image at a time so we can just take the first element
        classification_results = classification_results[0]
        class_names = classification_results.names
        speed = classification_results.speed
        
        probabilities = classification_results.probs.cpu()
        classification = class_names[probabilities.top1]
        confidence = float(probabilities.top1conf)
        
        return classification, confidence, speed