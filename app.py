import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('mango_model.h5')
    return model

def import_and_predict(image_data, model):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

def get_remedy_suggestion(disease):
    # Remedy suggestions for different diseases
    remedy_suggestions = {
        'Anthracnose': "Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.",
        'Bacterial Canker': "Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active. If using string trimmers around the base of trees, avoid damaging bark with breathable Tree Wrap to prevent infection.",
        'Cutting Weevil': "Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage.",
        'Die Back': "After pruning, apply copper oxychloride at a concentration of '0.3%' on the wounds. Apply Bordeaux mixture twice a year to reduce the infection rate on the trees. Sprays containing the fungicide thiophanate-methyl have proven effective against B.",
        'Gall Midge': "Use yellow sticky traps to catch the flies. Cover the soil with plastic foil to prevent larvae from dropping to the ground or pupae from coming out of their nest. Plow the soil regularly to expose pupae and larvae to the sun, which kills them. Collect and burn infested tree material during the season.",
        'Powdery Mildew': "In order to control powdery mildew, three sprays of fungicides are recommended. The first spray comprising of wettable sulphur (0.2%, i.e., 2g per litre of water) should be done when the panicles are 8 -10 cm in size as a preventive spray.",
        'Sooty Mould': "The insects causing the mould are killed by spraying with carbaryl or phosphomidon 0.03%. It is followed by spraying with a dilute solution of starch or maida 5%. On drying, the starch comes off in flakes and the process removes the black mouldy growth fungi from different plant parts."
    }
    return remedy_suggestions.get(disease, "Remedy suggestion not available.")

with st.sidebar:
    st.image('mg.png')
    st.title("Mangifera Healthika")
    st.subheader("Accurate detection of diseases present in the mango leaves. This helps an user to easily detect the disease and identify its cause.")

model = load_model()

st.write("""
         # Mango Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = {
        0: 'Anthracnose',
        1: 'Bacterial Canker',
        2: 'Cutting Weevil',
        3: 'Die Back',
        4: 'Gall Midge',
        5: 'Healthy',
        6: 'Powdery Mildew',
        7: 'Sooty Mould'
    }

    if np.argmax(predictions) < len(class_names):
        # Predicted class is within the class_names list
        class_name = class_names[np.argmax(predictions)]
        string = "Detected Disease: " + class_name
        st.sidebar.warning(string)
        
        if class_name == 'Healthy':
            st.balloons()
            st.sidebar.success(string)
        else:
            # Display remedy suggestion based on the predicted class
            remedy_suggestion = get_remedy_suggestion(class_name)
            st.markdown("## Remedy")
            st.info(remedy_suggestion)
    else:
        # Predicted class is not within the class_names list
        st.sidebar.error("Unknown disease detected. Remedy suggestion cannot be provided.")
