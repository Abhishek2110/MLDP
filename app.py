import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('mango_model.h5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Mango Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def import_and_predict(image_data, model, true_class=None):
    try:
        size = (224, 224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        
        if true_class:
            accuracy = 1 if true_class == predicted_class else 0
        else:
            accuracy = None
        
        return predicted_class, accuracy
    except Exception as e:
        st.error("Error predicting image: {}".format(str(e)))

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    true_class = st.text_input("Enter the true class of the uploaded image (if known):")
    
    predicted_class, accuracy = import_and_predict(image, model, true_class)
    
    if predicted_class == 'Healthy':
        st.balloons()
        st.sidebar.success("Detected Disease: Healthy")
    else:
        if predicted_class in class_names:
            st.sidebar.warning("Detected Disease: " + predicted_class)
            st.markdown("## Remedy")

            # Add remedy suggestions based on predicted disease
            if predicted_class == 'Anthracnose':
                st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")
            elif predicted_class == 'Bacterial Canker':
                st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")
            # Add other disease remedies here...
        else:
            st.error("Incorrect image or not a mango leaf image. Please upload a valid mango leaf image.")
    
    if accuracy is not None:
        st.sidebar.error("Accuracy : {} %".format(accuracy * 100))
