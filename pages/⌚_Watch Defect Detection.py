import streamlit as st
from streamlit_modal import Modal
from streamlit_image_select import image_select
import os
from model_inference.model_pipelines import results_to_df
import cv2

# Set Streamlit page configuration
st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

# Copilot Gifs
copilot_gifs = {"Big Ring" : {"Reference": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Big%20Ring%20-%20Reference%20-%20v4.gif",
                              "Good": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Big%20Ring%20-%20Good%20-%20v4.gif",
                              "Bad": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Big%20Ring%20-%20Bad%20-%20v4.gif"},
                "Small Ring" : {"Reference": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Small%20Ring%20-%20Reference%20-%20v4.gif",
                                "Good": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Small%20Ring%20-%20Good%20-%20v4.gif",
                                "Bad": "https://storage.cloud.google.com/auto-ai_resources_fo/asher_model_export/Watch%20Copilot%20Assets/Small%20Ring%20-%20Bad%20-%20v4.gif"}}

# Function to check if models are configured
def check_models_configurated():
    """
    Checks if the models are configured.

    Returns:
        bool: True if models are configured, False otherwise.
    """
    # Check if configuration is complete in session state
    if "configuration_complete" not in st.session_state:
        st.session_state.configuration_complete = False
        return False
    elif st.session_state["configuration_complete"] == False:
        return False
    elif st.session_state["configuration_complete"] == True:
        return True

# Function to check if GCS authentication is done
def check_for_GCS_authentication():
    """
    Checks if Google Cloud Storage (GCS) authentication is done.

    Returns:
        bool: True if GCS authentication is done, False otherwise.
    """
    # Check if GCS authentication is done in session state
    if "GCS_Authenticated" not in st.session_state:
        st.session_state.GCS_Authenticated = False
        return False
    elif st.session_state["GCS_Authenticated"] == False:
        return False
    elif st.session_state["GCS_Authenticated"] == True:
        return True

# Function to check if defect detection pipeline is available
def check_for_defect_detection_pipeline():
    """
    Checks if the defect detection pipeline is available.

    Returns:
        bool: True if defect detection pipeline is available, False otherwise.
    """
    # Check if defect detection pipeline is available in session state
    if "defect_detection_pipeline" not in st.session_state:
        st.session_state.defect_detection_pipeline = None
        return False
    elif st.session_state["defect_detection_pipeline"] == None:
        return False
    elif st.session_state["defect_detection_pipeline"] != None:
        return True

# Function to initialize session state variables
def state_initialization():
    """
    Initializes the session state variables.
    """
    # Initialize model inference results
    if "model_inference_results" not in st.session_state:
        st.session_state["model_inference_results"] = {}

    # Initialize inference mode
    if "inference_mode" not in st.session_state:
        st.session_state["inference_mode"] = False

    # Initialize inspection mode
    if "inspection_mode" not in st.session_state:
        st.session_state["inspection_mode"] = False

    # Initialize selected image
    if "selected_image" not in st.session_state:
        st.session_state["selected_image"] = None

    if "detected_object_images" not in st.session_state:
        st.session_state["detected_object_images"] = {} 

# Function to verify setup
def setup_verification():
    """
    Verifies the setup by checking if models are configured, GCS authentication is done, and defect detection pipeline is available.

    Returns:
        bool: True if setup is verified, False otherwise.
    """
    if check_models_configurated() and check_for_defect_detection_pipeline() and check_for_GCS_authentication():
        state_initialization()
        return True
    else:
        return False
    
def store_cropped_images(defect_detection_results):
    """
    Stores the cropped images in the session state.

    Args:
        defect_detection_results (dict): The defect detection results.
    """
    

    

# Function to show image selections
def show_image_selections(container):
    """
    Shows the image selections.

    Args:
        container (streamlit.container.Container): Streamlit container to display the image selections.

    Returns:
        str: The selected image path.
    """
    base_path = os.path.join("pages", "sample_images")
    
    with container:
        img = image_select(label="Select an image", images=[os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith((".jpg", ".png", ".jpeg"))])
        return img

# Function to show cropped image selections
def show_cropped_image_selections(container, input_image):
    """
    Shows the cropped image selections.

    Args:
        container (streamlit.container.Container): Streamlit container to display the cropped image selections.
        input_image (str): The path of the original input image.

    Returns:
        str: The selected cropped image path.
    """
    
    defect_detection_results = st.session_state["model_inference_results"][input_image][0]

    # for part, data in defect_detection_results.items():
    #     print(data, end="\n\n")


    with container:

        col1, col2, _ = container.columns([1, 1, .25])

        with col1:

            img = image_select(label="Select a part to inspect", images=[data["Cropped Image Array"] for data in defect_detection_results.values()],
                            captions=[f"Part: {part}" for part in defect_detection_results.keys()], return_value="index", use_container_width=False)
            
        with col2:
            #print(defect_detection_results["Big Gear"])
            print(defect_detection_results[list(defect_detection_results.keys())[img]]["Cropped Image Array"], list(defect_detection_results.keys())[img])
            st.image(defect_detection_results[list(defect_detection_results.keys())[img]]["Cropped Image Array"], use_column_width=True)

            return img

# Function for model inference
def model_inference(input_image, return_df=False):
    """
    Performs model inference on the input image.

    Args:
        input_image (str): The path of the input image.
        return_df (bool, optional): Whether to return the results as a pandas DataFrame. Defaults to False.

    Returns:
        tuple: A tuple containing the defect detection results, object detection speed, and classification speed.
    """
    defect_detection_pipeline = st.session_state["defect_detection_pipeline"]

    defect_detection_results, object_detection_speed, classification_speed = defect_detection_pipeline.detect_defects(input_image=input_image, 
                                                                                                                      confidence_threshold=0.30, 
                                                                                                                      iou_threshold=0.30, 
                                                                                                                      device="cpu", 
                                                                                                                      excluded_parts=st.session_state["excluded_objects"],
                                                                                                                      return_df=return_df)

    return defect_detection_results, object_detection_speed, classification_speed

# Function for state-managed model inference
def state_managed_model_inference(input_image, return_df=False):
    """
    Performs state-managed model inference on the input image.

    Args:
        input_image (str): The path of the input image.
        return_df (bool, optional): Whether to return the results as a pandas DataFrame. Defaults to False.

    Returns:
        tuple: A tuple containing the defect detection results, object detection speed, and classification speed.
    """
    already_processed_images = list(st.session_state["model_inference_results"].keys())

    if input_image in already_processed_images:
        defect_detection_results, object_detection_speed, classification_speed = st.session_state["model_inference_results"][input_image]
    else:
        defect_detection_results, object_detection_speed, classification_speed = model_inference(input_image, return_df=return_df)
        st.session_state["model_inference_results"][input_image] = (defect_detection_results, object_detection_speed, classification_speed)

    return defect_detection_results, object_detection_speed, classification_speed, input_image

# Function to display bounding boxes on the image
def display_bounding_boxes(image, results):
    """
    Displays bounding boxes on the image based on the defect detection results.

    Args:
        image (str or numpy.ndarray): The path or numpy array of the image.
        results (dict): The defect detection results.

    Returns:
        numpy.ndarray: The image with bounding boxes.
    """
    for label, data in results.items():
        bounding_box = data['Bounding Box']
        classification = data['Classification']

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, bounding_box)

        # Draw bounding box on the image
        if classification in ["Non Defective", "Excluded"]:
            color = (0, 255, 0)  # Green color for non-defective
        else:
            color = (255, 0, 0)  # Red color for defective

        if isinstance(image, str):
            image = cv2.imread(image)

        text = f"{label}: {classification}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    return image

# Function for recoloring defective cells in the results table
def defective_recoloring(val):
    """
    Recolors the defective cells in the results table.

    Args:
        val (str): The cell value.

    Returns:
        str: The CSS style for the cell background color.
    """
    color = 'red' if val != "Non Defective" else 'green'
    return f'background-color: {color}'

def show_copilot_modal(selected_cropped_image, part_name):
    modal = Modal(title="Individual Part Analysis", key="Copilot Modal", max_width=1400)

    with modal.container():
        left_side, right_side = st.columns([.6, .4])


        if part_name != "Big Gear":
            with left_side:

                st.markdown(f"<h2 style='text-align: center;'>This is a cropped version of the {part_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Use the reference guide to the right to better understand this image.</p>", unsafe_allow_html=True)

                _, center_col, _ = st.columns([1, 3, 1])

                with center_col:
                    st.image(selected_cropped_image, use_column_width=True)

            with right_side:
                st.subheader(f"Reference Guide for {part_name}s")
                for key, value in copilot_gifs[part_name].items():
                    st.image(value, caption=key, use_column_width=True)
        else:
            st.markdown(f"<h2 style='text-align: center;'>This is a cropped version of the {part_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>The Big Gear is a special case and does not have a reference guide.</p>", unsafe_allow_html=True)
            
            _, center_col, _ = st.columns([1, 1, 1])

            with center_col:
                st.image(selected_cropped_image, use_column_width=True)

        st.divider()

        close_model_button = st.button(label="Close", use_container_width=True, type="primary")
        if close_model_button:
            st.session_state["inspection_mode"] = False
            st.rerun()

# Function for the image selection view
def image_selection_view(image_container):
    """
    Displays the image selection view.

    Args:
        image_container (streamlit.container.Container): Streamlit container to display the image selection view.
    """
    image = show_image_selections(image_container)
        
    _, col2, _ = image_container.columns([1, 1, 1])

    col2.image(image, caption=f"Image Name: {image.split('/')[-1]}", use_column_width=True)
    defect_detection_button = image_container.button(label="Detect for Defects", use_container_width=True, type="primary")
    if defect_detection_button:
        st.session_state["inference_mode"] = True
        st.session_state["selected_image"] = image
        st.rerun()

# Function for the defect detection view
def defect_detection_view(image_container):
    """
    Displays the defect detection view.

    Args:
        image_container (streamlit.container.Container): Streamlit container to display the defect detection view.
    """

    image = st.session_state["selected_image"]

    defect_detection_results, object_detection_speed, classification_speed, input_image = state_managed_model_inference(image)

    left_side_container, _, right_side_container = image_container.columns([.49, .02, .49])

    annotated_image = display_bounding_boxes(image, defect_detection_results)
    left_side_container.image(annotated_image, caption=f"Image Name: {image.split('/')[-1]}")
    
    results_df = results_to_df(defect_detection_results)

    results_df.rename(columns={"Classification_Confidence": "Confidence"}, inplace=True)

    # Fill in missing parts if there are any
    detected_parts = set(results_df.index)
    all_parts = set(st.session_state["object_detection_model"].names)
    missing_parts = all_parts - detected_parts

    if missing_parts:
        missing_parts_df = results_to_df({part: {"Bounding Box": None, "Classification": "Missing", "Confidence": 1.0} for part in missing_parts})
        results_df = results_df._append(missing_parts_df)

    right_side_container.subheader("Detailed Report")
    right_side_container.table(results_df.style.map(defective_recoloring, subset=["Classification"]))
    right_side_container.divider()

    left_col, right_col = right_side_container.columns([.5, .5])

    left_col.metric(label="Object Detection Speed", value=f"{object_detection_speed['inference']:.2f}ms 💨")
    right_col.metric(label="Overall Classification Speed", value=f"{classification_speed:.2f}ms 💨")
    right_side_container.divider()

    select_cropped_image_index = show_cropped_image_selections(right_side_container, input_image)
    selected_cropped_image = defect_detection_results[list(defect_detection_results.keys())[select_cropped_image_index]]["Cropped Image Array"]
    selected_cropped_image_part_name = list(defect_detection_results.keys())[select_cropped_image_index]

    # _, select_cropped_image_col, _ = right_side_container.columns([1, .5, 1])

    # select_cropped_image_col.image(selected_cropped_image, use_column_width=True)

    inspect_button = right_side_container.button(label="Inspect", use_container_width=True, type="primary")

    if inspect_button:
        st.session_state["inspection_mode"] = True
        show_copilot_modal(selected_cropped_image, selected_cropped_image_part_name)

    back_button = image_container.button(label="Run on Another Image", use_container_width=True, type="primary")
    if back_button:
        st.session_state["inference_mode"] = False
        st.rerun()

# Main function
def main():
    """
    Main function to run the Watch Defect Detection Streamlit App.
    """
    st.subheader("Defect Detection")

    configured_and_ready = setup_verification()

    image_container = st.container()

    if not configured_and_ready:
        st.write("Models aren't configurated, please go to the configuration page to configure the models")
        return

    if not st.session_state["inference_mode"]:
        image_selection_view(image_container)
    else:
        defect_detection_view(image_container)
    
if __name__ == "__main__":
    main()