import streamlit as st
from PIL import Image
from model_inference.model_wrappers import YOLOModelWrapper
from model_inference.model_pipelines import DefectDetectionPipeline
import os
import cv2
import numpy as np
from stqdm import stqdm

# TODO: DONE Make an initializer for easier model creation because this is too much code DONE

# def initialize_models(GCS_bucket_name:str, GCS_model_directory:str, model_name_and_file_name_dict: dict, GCS_Auth_JSON_Path:str, local_model_storage_dir:str, model_type:str):
#     models = {}
#     for model_name, model_file_name in model_name_and_file_name_dict.items():
#         model = YOLOModelWrapper(GCS_bucket_name=GCS_bucket_name,
#                                 GCS_model_directory=GCS_model_directory,
#                                 GCS_model_file_name=model_file_name,
#                                 GCS_Auth_JSON_Path=GCS_Auth_JSON_Path,
#                                 local_model_storage_dir=local_model_storage_dir,
#                                 model_type=model_type)
#         models[model_name] = model
#     return models

# all_models = initialize_models(GCS_bucket_name="auto-ai_resources_fo",
#                                GCS_model_directory="asher_model_export/watch-defect-models",
#                                 model_name_and_file_name_dict={"Object Detection": "watch-segmentation-model.pt",
#                                                                 "Big Ring": "watch-big_ring-classification-model.pt",
#                                                                 "Small Ring": "watch-small_ring-classification-model.pt"},
#                                 GCS_Auth_JSON_Path=os.path.join(os.getcwd(), "faceopen_key.json"),
#                                 local_model_storage_dir=None,
#                                 model_type=None)

# object_detection_model = all_models["Object Detection"]
# classification_models = {"Big Ring": all_models["Big Ring"], "Small Ring": all_models["Small Ring"]}
# defect_detection_pipeline = DefectDetectionPipeline(object_detection_model, classification_models)



# def draw_bounding_boxes(image, results):
#     for label, data in stqdm(results.items()):
#         bounding_box = data['Bounding Box']
#         classification = data['Classification']

#         # Convert bounding box coordinates to integers
#         x1, y1, x2, y2 = map(int, bounding_box)

#         # Draw bounding box on the image
#         if classification in ["Non Defective", "Excluded"]:
#             color = (0, 255, 0)  # Green color for non-defective
#         else:
#             color = (255, 0, 0)  # Red color for defective

#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

#     return image

st.set_page_config(page_title="Home", page_icon="🏠")

def main():
    st.title("Welcome to Watch Defect Detection App")
    st.write("This app allows you to detect defects in watches using computer vision techniques.")
    
    # Navigation
    pages = {
        "Home": home_page,
        "Defect Detection": defect_detection_page,
        "About": about_page
    }
    
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    pages[page]()
    
def home_page():
    st.write("This is the home page.")
    # Add content for the home page here
    
def defect_detection_page():
    st.write("This is the defect detection page.")
    # Add content for the defect detection page here
    
def about_page():
    st.write("This is the about page.")
    # Add content for the about page here
    
if __name__ == "__main__":
    main()

