import streamlit as st
from streamlit_image_select import image_select
import os
from PIL import Image

# Plan for what this page's flow to do

"""
Note: Try to add animations and make a mockup of the page to see how it will look
in figma or something similar

Note: Add Text to Make everything completely self explanatory

1. Verify if model's are configurated
    1a. If they aren't display a graphic saying it's not configurated
    which will feature a button or something clickable that will redirect
    to the configuration screen to solve this
    1b. If they are configurated put a notice in a small area saying
    the models are configured and show the diagram of the model flow
2. Show a couple of images to show from or upload your own xray watch image
3. After Uploading show a live graphic of th model's flow
4. After model inference show the results
    4a. First results should be the model's bounding box annotations on the
    original image
        4a1. Make the bounding boxes clickable, if clicked it will show the
        cropped image and the classification results with the reference guide
    4b. Second results should be the model's classification results
"""

# List of session state variables
# - object_detection_model_count
# - classification_model_count
# - object_detection_model
# - object_detection_model_name
# - defect_detection_pipeline
# - classification_models
# - excluded_objects
# - GCS_Authenticated
# - configuration_complete

def check_models_configurated():
    if "configuration_complete" not in st.session_state:
        st.session_state.configuration_complete = False
        return False
    elif st.session_state["configuration_complete"] == False:
        return False
    elif st.session_state["configuration_complete"] == True:
        return True
    
def check_for_GCS_authentication():
    if "GCS_Authenticated" not in st.session_state:
        st.session_state.GCS_Authenticated = False
        return False
    elif st.session_state["GCS_Authenticated"] == False:
        return False
    elif st.session_state["GCS_Authenticated"] == True:
        return True

def check_for_defect_detection_pipeline():
    if "defect_detection_pipeline" not in st.session_state:
        st.session_state.defect_detection_pipeline = None
        return False
    elif st.session_state["defect_detection_pipeline"] == None:
        return False
    elif st.session_state["defect_detection_pipeline"] != None:
        return True
    
def setup_verification():
    if check_models_configurated() and check_for_defect_detection_pipeline() and check_for_GCS_authentication():
        return True
    else:
        return False
    
def show_image_selections():
    base_path = "images"

    img = image_select(label="Select an image", images=[os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith((".jpg", ".png", ".jpeg"))])
    return img

def show_model_flow():
    pass

def show_upload_image():
    pass

def show_model_inference():
    pass

def show_results():
    pass

def main():
    configured_and_ready = setup_verification()

    if configured_and_ready:
        show_image_selections()

if __name__ == "__main__":
    main()