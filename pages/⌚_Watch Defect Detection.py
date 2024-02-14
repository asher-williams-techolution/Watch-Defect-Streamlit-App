import streamlit as st
from streamlit_modal import Modal
from streamlit_image_select import image_select
import os
from model_inference.model_pipelines import results_to_df
import cv2

# Set Streamlit page configuration
st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

# Copilot Gifs
copilot_gifs = {"Big Ring" : {"Reference": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b609eebc914988ccfb3/referenceDataGroups/referenceDataGroup-65b19d123965bf6264a0ebc8/referenceDatas/referenceData-65b3f836a612d53e2705069c/Big%20Ring%20-%20Reference%20-%20v2.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192513&Signature=abrTIdvGPgQHJvvGmfsNAsnBoycmiAQH%2FKCF3nkuS7vs%2F7xjHYXjt0UnMtXeoJ05DJaar1AzcChzmWZ9obRsetII7CMF%2FJVB9f63suyyjqvy5Ry80Lo3fY1oI7g3AqCiUfnTLqsildL3oFf%2Fug20DIIDHfbAd%2BYcJqC3nB4zR0RHm60y9XdD60k02Sm%2FfSgD9YIoJhJHxkQD1vnhXmZvno3j28IN5dlbsPKvVf%2BaxYe3HVxLPYHDEEzpogEh%2BrI92G4jg3r5iuZKu9begKSUSw%2BjdY1CeLVu9l4rW7jtFeYkyU%2BLNWXAv1gLpR8DKnVIVaH9Wj63pi5PJ2BPcj64hQ%3D%3D",
                              "Good": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b609eebc914988ccfb3/referenceDataGroups/referenceDataGroup-65b19d153965bf6264a0ebc9/referenceDatas/referenceData-65b19d38a528649e70f5e9a4/Big%20Ring%20-%20Good.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192513&Signature=G7sy4ZZu8pIFyGfgfroH8A%2Bj1CAwiI2IoxHwT0Q1mlimVKWycCEjG4Ryi8cppXv8PjSoraUeM1ehGYQtnbXslMRRKIEat%2B%2BpZY6TShLVXlweZCCT5Z9FfLMpy10Y%2Bugw7bwr1ruQ%2BQh%2FkVApSssIsUlY89kmVUAwZc2uWJF%2B0UYlFgBrmw37zd4An3Yer5U4%2F%2B7oHmmCMJa%2FqW%2F4avGoIrg%2BgjkL%2F6SaMYJeX6KYVoR6%2F88Z8hb2wwbaq67Z4kCS2GYmNAEBZVcZhVvozVaJ7yrdD9Bi7zRLnvbe70E%2Fnlyg7MB74JGYdGarPZ8M2uZEp92PXBQ%2BsuyoYkc1fl8XkA%3D%3D",
                              "Bad": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b609eebc914988ccfb3/referenceDataGroups/referenceDataGroup-65b19d153965bf6264a0ebca/referenceDatas/referenceData-65b19d38a528649e00f5e9a5/Big%20Ring%20-%20Bad.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192513&Signature=aExPan9j2SdJSKgOrdnINy4xEx2pPJLOwMBIxMVg8BwsVoIU1FxL6YmTDtgWxPgkDPSkxajxPmOdyIJvvy2yUedLFSQTibWp0uLQqRv%2B2%2FtTlgKQsV3GdQrQJD5mX8IJ%2BPXzfKK9EyPmhFA8JAkAZYS5TM8D5hy9pTXHgWRWv7E7mdnrOPIZAozXvMDNFlFf3qy7JrGJDFDVUCJvrRthuFNvSh9TPxuJwfYiNNUA13Yrp2BgV9nV3v%2BVrGdSxoaqfi845vC0YeQh64h8hm3vOlkcZMO5vgt%2FuVEVLJKWS%2BIH4fRy7ZB78P9n16UepJaXvGx35EVKv7pmDSTn40pYOA%3D%3D"},
                "Small Ring" : {"Reference": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b689eebc956628cd35b/referenceDataGroups/referenceDataGroup-65b19d443965bf6264a0ebcb/referenceDatas/referenceData-65b3f856a612d5739b051248/Small%20Ring%20-%20Reference%20-%20v2.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192912&Signature=B5qcUu5uXOyrhxlH8LCvB6nkYk5wW198Ws5aqjEBulYgNLSsKvMTWgSMhPtBysURrs5awLMDn1vS7ngzwqmOTjjL0lpk%2BJqP8MU5mbzXPbt4w0xOErSjZCzu6hLITnSDbKJ7JBsxPr8a1%2FdboOS391KRSZYFfQFJ%2B6%2Fuq%2BY2JxCDi%2Bv%2FH2VrHz2I5kMsrK%2B4Ut6TdYR6wEYSp5OiDlbQDEwZj5MBkZpJvERvzuhcNr9kn5EXpNfIIchd81IpLzLmmr1MI9gJO1q3gsRvG%2B8R8cH6dU9Ch9CxntnMY4mSItL4gQg41F32f8Hmn0oqateDr7V9uWTg4wtqnu7Ibxj5sQ%3D%3D",
                                "Good": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b689eebc956628cd35b/referenceDataGroups/referenceDataGroup-65b19d453965bf6264a0ebcc/referenceDatas/referenceData-65b19d64a528645c5af5efa1/Small%20Ring%20-%20Good.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192912&Signature=JYivf94clAoiHqKo6XBjLm6EV0wAHsKF6M1srkvAHIxG1fBXpR88wFXGjPm6WnGk%2Fks5tnKZoSG9%2BRh%2F4roNqR%2Fuu95sGhJftNfxhyaqSwJyK4yYgzyrD2lBohZMQcsKSHMJ0cnKjDClJRNbL%2BCpG1fFSbSVRGdI5GUX4nhIt84c9fNHgaraCvGH0qy1Vt%2F7WpevJiJQT48aZEWURbqQGP%2F4p2Kdwh%2BQ8rDAGFL99XLc%2FM39%2B%2FV%2FrEpqxwslFMpA47LQxrQMc4QTn1K9i85vF92ehoqMSxZXdWK2gbpE7F%2BC0oVhYuVPzc3uXefbHOuFICzXYEZBHzPkOrFgWNpaWQ%3D%3D",
                                "Bad": "https://storage.googleapis.com/auto-ai_resources_fo/project-6530fd75cae09dbf9b95d504/copilot-65af4b689eebc956628cd35b/referenceDataGroups/referenceDataGroup-65b19d453965bf6264a0ebcd/referenceDatas/referenceData-65b19d64a528644d0ff5efa2/Small%20Ring%20-%20Bad.gif?GoogleAccessId=rlef-cloud-storage-admin%40faceopen-techolution.iam.gserviceaccount.com&Expires=1708192912&Signature=CVHi4pyonbDFSu%2FYnTkwnQ%2B6VyUSlGCH9FRTEUUf5%2FDaPB1IdqrvBewCd8uoSlqo8SSjIyYD6KFfUUwoHW3ORTDJglxgLhaePFY4iR%2Fv9n54cvvhREwGEBKuI0JPCC%2F4KnMz9tpNDEnV9%2B7Z0axVpOE3uyz3dgydfYOUKCB1e3LKDZem9fv7s%2FMpdZShd%2BuLGSUGxdSGX9qtQZ0KO%2BBjIwexpftdiDaR4TbmeEKS84sYmAW619fJcKFB4lPRuYsWIN4DnpvAh4yqWsqwnNOdMhAvuul%2Fi5LIM%2FtCi%2FQ%2F%2BqbtkljtmQYjdiBAh2DeEaRo8EBSR%2FkrzVmDhABTmKhN%2FA%3D%3D"}}

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
def show_cropped_image_selections(container):
    """
    Shows the cropped image selections.

    Args:
        container (streamlit.container.Container): Streamlit container to display the cropped image selections.

    Returns:
        str: The selected cropped image path.
    """
    base_path = st.session_state["object_detection_model"].local_image_storage_dir

    with container:
        img = image_select(label="Select a part to inspect", images=[os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith((".jpg", ".png", ".jpeg"))],
                           captions=[file.split("_cropped")[0] for file in os.listdir(base_path) if file.endswith((".jpg", ".png", ".jpeg"))])
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
                                                                                                                      confidence_threshold=0.60, 
                                                                                                                      iou_threshold=0.60, 
                                                                                                                      device="mps", 
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

    return defect_detection_results, object_detection_speed, classification_speed

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
    color = 'red' if val!="Non Defective" else 'green'
    return f'background-color: {color}'

def show_copilot_modal(selected_cropped_image):
    modal = Modal(title="Individual Part Analysis", key="Copilot Modal", max_width=1400)

    part_name = selected_cropped_image.split("_cropped")[0].split("/")[-1]
    print(part_name)

    with modal.container():
        left_side, right_side = st.columns([.6, .4])


        if part_name != "Big Gear":
            with left_side:

                st.markdown(f"<h2 style='text-align: center;'>This is a cropped version of the {part_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Use the reference guide to the right to better understand this image.</p>", unsafe_allow_html=True)

                _, center_col, _ = st.columns([1, 3, 1])

                with center_col:
                    st.image(selected_cropped_image, caption=f"Image Name: {selected_cropped_image.split('/')[-1]}", use_column_width=True)

            with right_side:
                st.subheader(f"Reference Guide for {part_name}s")
                for key, value in copilot_gifs[part_name].items():
                    st.image(value, caption=key, use_column_width=True)
        else:
            st.markdown(f"<h2 style='text-align: center;'>This is a cropped version of the {part_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>The Big Gear is a special case and does not have a reference guide.</p>", unsafe_allow_html=True)
            
            _, center_col, _ = st.columns([1, 1, 1])

            with center_col:
                st.image(selected_cropped_image, caption=f"Image Name: {selected_cropped_image.split('/')[-1]}", use_column_width=True)

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

    col2.image(image, caption=f"Image Name: {image.split('/')[-1]}", width=800)
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

    defect_detection_results, object_detection_speed, classification_speed = state_managed_model_inference(image)

    left_side_container, _, right_side_container = image_container.columns([.49, .02, .49])

    annotated_image = display_bounding_boxes(image, defect_detection_results)
    left_side_container.image(annotated_image, caption=f"Image Name: {image.split('/')[-1]}")
    
    results_df = results_to_df(defect_detection_results)

    results_df.rename(columns={"Classification_Confidence": "Confidence"}, inplace=True)

    right_side_container.subheader("Detailed Report")
    right_side_container.table(results_df.style.map(defective_recoloring, subset=["Classification"]))
    right_side_container.divider()

    left_col, right_col = right_side_container.columns([.5, .5])

    left_col.metric(label="Object Detection Speed", value=f"{object_detection_speed['inference']:.2f}ms 💨")
    right_col.metric(label="Overall Classification Speed", value=f"{classification_speed:.2f}ms 💨")
    right_side_container.divider()

    select_cropped_image = show_cropped_image_selections(right_side_container)

    _, select_cropped_image_col, _ = right_side_container.columns([1, .5, 1])

    select_cropped_image_col.image(select_cropped_image, use_column_width=True)

    inspect_button = right_side_container.button(label="Inspect", use_container_width=True, type="primary")

    if inspect_button:
        st.session_state["inspection_mode"] = True
        show_copilot_modal(select_cropped_image)

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