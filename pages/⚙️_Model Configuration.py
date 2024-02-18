import streamlit as st
import os
from model_inference.model_wrappers import YOLOModelWrapper
from model_inference.model_pipelines import DefectDetectionPipeline
from google.cloud import storage
from google.oauth2 import service_account
import networkx as nx
import tempfile
import matplotlib.pyplot as plt
import json

# TODO: Add a locking mechanism to prevent modification

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

default_bucket_name = "auto-ai_resources_fo"
default_model_directory = "asher_model_export/watch-defect-models"
default_GCS_Auth_JSON_Path = os.path.join(os.getcwd(), "faceopen_key.json")
default_object_detection_model_file_name = "watch-segmentation-model.pt"
default_cls_model_file_name = {"Big Ring": "watch-big_ring-classification-model.pt",
                               "Small Ring": "watch-small_ring-classification-model.pt"}


def session_state_initialization():
    if "object_detection_model_count" not in st.session_state:
        st.session_state.object_detection_model_count = 0

    if "classification_model_count" not in st.session_state:
        st.session_state.classification_model_count = 0

    if "classification_models" not in st.session_state:
        st.session_state.classification_models = {}

    if "excluded_objects" not in st.session_state:
        st.session_state.excluded_objects = ["Big Gear"]

    if "GCS_Authenticated" not in st.session_state:
        st.session_state.GCS_Authenticated = True if check_for_gcs_authentication() else False

    if "configuration_complete" not in st.session_state:
        st.session_state.configuration_complete = False

def authenticate_gcs(json_path=None):
    try:
        if json_path:
            # Read the file and convert it to a JSON string
            with open(json_path) as json_file:
                credentials_info = json.load(json_file)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            storage.Client(credentials=credentials)
        else:
            # Use credentials from st.secrets
            credentials_info = st.secrets["gcs"]
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            storage.Client(credentials=credentials)
        
        # Store the credentials in session state
        st.session_state['gcs_credentials'] = credentials
        st.session_state.GCS_Authenticated = True
        return True
    except Exception as e:
        st.error(f"Authentication Failed: {e}")
        return False

def check_for_gcs_authentication():
    try:    
        authenticate_gcs()
    except:
        return False
 
def check_for_obj_detection_model():
    if "object_detection_model" in st.session_state:
        return True
    else:
        return False

def check_if_classification_models_added():
    valid_part_names = set(st.session_state["object_detection_model"].names)
    excluded_objects = set(st.session_state["excluded_objects"])
    already_created_models = set(st.session_state.classification_models.keys())
    available_part_names = valid_part_names - excluded_objects - already_created_models
    if len(available_part_names) == 0:
        return True
    else:
        return False
    
def validate_file_name(file_name:str):
    existing_file_names = [model.GCS_model_file_name for model in st.session_state.classification_models.values()]

    if file_name in existing_file_names:
        return False
    else:
        return True

    
def check_if_model_pipeline_created():
    if "defect_detection_pipeline" in st.session_state:
        st.session_state["configuration_complete"] = True
        return True
    else:
        return False

def authentication_form():
    if not check_for_gcs_authentication():
        form = st.form("Authentication Form")
        json_path = form.text_input("Please enter the path to your GCS Authentication JSON")
        if form.form_submit_button("Authenticate"):
            if authenticate_gcs(json_path):
                form.success("Authenticated")
                st.rerun()
            else:
                form.error("Failed to Authenticate")
    else:
        st.success("GCS Authenticated")

def add_object_detection_model():

    if "object_detection_model" in st.session_state:
        st.subheader("Object Detection Model Already Added")
        object_detection_model = st.session_state["object_detection_model"]
        object_detection_model_name = st.session_state["object_detection_model_name"]
        st.info("Object Detection Model Name: " + object_detection_model_name)
    else:
        st.subheader("Add an Object Detection Model")
        obj_model_form = st.form("Object Detection Model Form")


        bucket_name = obj_model_form.text_input(label="GCS Bucket Name", value=default_bucket_name if "GCS Bucket Name" not in st.session_state else st.session_state["GCS Bucket Name"]) 
        directory = obj_model_form.text_input(label="GCS Directory", value=default_model_directory if "GCS Directory" not in st.session_state else st.session_state["GCS Directory"]) 
        file_name = obj_model_form.text_input(label="File Name", value=default_object_detection_model_file_name if "OBJ MODEL GCS File Name" not in st.session_state else st.session_state["OBJ MODEL GCS File Name"]) 
        model_name = obj_model_form.text_input(label="Model Name", value="" if "object_detection_model_name" not in st.session_state else st.session_state["object_detection_model_name"]) 

        if obj_model_form.form_submit_button("Add Model"):
            object_detection_model = YOLOModelWrapper(GCS_bucket_name=bucket_name,
                                                      GCS_model_directory=directory,
                                                      GCS_model_file_name=file_name,
                                                      GCS_Auth_JSON_Path=None,
                                                      local_model_storage_dir=None,
                                                      model_type=None)
            st.session_state["object_detection_model"] = object_detection_model
            st.session_state["object_detection_model_name"] = model_name
            st.session_state["object_detection_model_count"] += 1
            st.session_state["GCS Bucket Name"] = bucket_name
            st.session_state["GCS Directory"] = directory
            st.session_state["OBJ MODEL GCS File Name"] = file_name

            obj_model_form.success("Object Detection Model Added")
            st.rerun()

def add_classification_model():

    # Display Valid Part Names
    possible_objects = set(st.session_state["object_detection_model"].names)
    excluded_objects = set(st.session_state["excluded_objects"])
    already_created_models = set(st.session_state.classification_models.keys())
    valid_part_names = possible_objects - excluded_objects - already_created_models

    # Check if valid part names is empty
    if len(valid_part_names) == 0:
        st.subheader("Classification Models Already Added")
        st.info(f"Models Added: {already_created_models}")
        st.success("All Valid Part Classification Models Have Been Added")
        return
    
    st.subheader("Add a Classification Model")

    st.write("Valid Part Names: ", valid_part_names)

    cls_model_form = st.form("Classification Model Form")

    bucket_name = cls_model_form.text_input(label="GCS Bucket Name", value=default_bucket_name if "GCS Bucket Name" not in st.session_state else st.session_state["GCS Bucket Name"]) 
    directory = cls_model_form.text_input(label="GCS Directory", value=default_model_directory if "GCS Directory" not in st.session_state else st.session_state["GCS Directory"]) 
    model_name = cls_model_form.selectbox(label="Model Name", options=list(valid_part_names))
    file_name = cls_model_form.text_input(label="File Name", value=default_cls_model_file_name[model_name])

    if cls_model_form.form_submit_button(label="Add Model", on_click=validate_file_name, args=(file_name,)):
            classification_model = YOLOModelWrapper(GCS_bucket_name=bucket_name,
                                                    GCS_model_directory=directory,
                                                    GCS_model_file_name=file_name,
                                                    GCS_Auth_JSON_Path=None,
                                                    local_model_storage_dir=None,
                                                    model_type=None)

            st.session_state.classification_models[model_name] = classification_model
            st.session_state.classification_model_count += 1
            cls_model_form.success("Classification Model Added")
            st.rerun()


def exclude_parts_from_object_detection():
    st.subheader("Exclude Parts from Object Classification")
    object_detection_model = st.session_state.object_detection_model
    possible_objects = object_detection_model.names
    excluded_objects = st.session_state.get("excluded_objects", [])
    excluded_objects = st.multiselect("Excluded Parts", possible_objects, default=excluded_objects)
    st.session_state["excluded_objects"] = excluded_objects


# Functionality Removed
# def validate_model_name(model_name:str):
#     possible_objects = st.session_state["object_detection_model"].names

#     if model_name in possible_objects:
#         return True
#     else:
#         return False
    
# def validate_model_name_callback(model_name:str):
#     if not validate_model_name(model_name):
#         st.error("Model Name not valid, must be a valid part name")
#         return False
#     else:
#         st.success("Model Name Valid")
#         return True

# TODO: Move this function to app_utils.visualization_utils and have it take customization parameters
def visualize_model_pipeline():

    object_detection_model_name = st.session_state.object_detection_model_name
    classification_models = st.session_state.classification_models

    plt.figure(figsize=(8, 8))
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add object detection model as the root node
    G.add_node(object_detection_model_name, shape='s', color='blue')

    # Add classification models as child nodes
    for model_name in classification_models:
        G.add_node(model_name, shape='s', color='green')
        G.add_edge(object_detection_model_name, model_name, label=model_name)
    
    # Set node positions for better visualization
    pos = {}
    num_classification_models = len(classification_models)

    # Position of classification models
    for i, model_name in enumerate(classification_models):
        pos[model_name] = (0 + i, 0)

    # Position of object detection model
    pos[object_detection_model_name] = ((num_classification_models - 1)*.5, 1)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, nodelist=[object_detection_model_name], node_color='lightblue', node_size=10000, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=classification_models.keys(), node_color='lightgreen', node_size=3000, node_shape='s', edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    # Remove axis and save the diagram as an image file
    plt.axis('off')
    plt.margins(0.1)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    plt.savefig(temp_file.name, format='png')
    plt.close()

    # Display the diagram as an image in the Streamlit app
    st.image(temp_file.name)

def create_model_pipeline():



    if "defect_detection_pipeline" not in st.session_state:
        with st.form("Model Pipeline Form"):
            st.subheader("Confirm that this is the correct model pipeline")
            visualize_model_pipeline()

            if st.form_submit_button("Confirm Model Pipeline"):
                object_detection_model = st.session_state.object_detection_model
                classification_models = st.session_state.classification_models
                defect_detection_pipeline = DefectDetectionPipeline(object_detection_model, classification_models)
                st.session_state["defect_detection_pipeline"] = defect_detection_pipeline
                st.success("Model Pipeline Created")
                st.rerun()
    else:
        st.subheader("Model Pipeline Configuration Diagram")
        visualize_model_pipeline()
        st.success("Model Pipeline Created")


def main():
    st.title("Model Configuration")

    st.divider()

    session_state_initialization()

    if not st.session_state.GCS_Authenticated:
        authentication_form()
    else:
        authentication_status = st.success("GCS Authenticated")

    st.divider()

    if check_for_gcs_authentication():
        add_object_detection_model()

    st.divider()

    if check_for_obj_detection_model():
        exclude_parts_from_object_detection()
        add_classification_model()

    st.divider()

    if check_for_obj_detection_model() and check_if_classification_models_added():
        create_model_pipeline()

    if check_if_model_pipeline_created():
        st.success("Model Configuration Complete! You can now perform Watch Defect Detection.")


if __name__ == "__main__":
    main()
