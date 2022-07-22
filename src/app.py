import streamlit as st
import tensorflow as tf
from PIL import Image
from const import food_list
import numpy as np

st.set_page_config(
    page_title="Food Sight",
    page_icon=":pizza",
    initial_sidebar_state="expanded",
    menu_items={"About": "# This is an *extremely* cool Food Sight app!"},
)


@st.cache(allow_output_mutation=True)
def Load_Food_Sight():
    model = tf.keras.models.load_model("/src/model/my_food_sight_model.h5")
    return model


def setup():
    with st.spinner("Food Sight is being loaded.."):
        model = Load_Food_Sight()

    st.write("""# Food Sight 🍕👀""")

    st.write(
        """Food Sight is an Image classification AI🤖 Web app that has been trained and fine tuned on top of the EfficientNetV2b0 Deep Neural network."""
    )  # description and instructions

    file = st.file_uploader(
        "Upload the image to be Food Sighted", type=["jpg", "png", "jpeg"]
    )
    # img_file_buffer = st.camera_input("Take a picture")
    return model, file


def upload_predict(upload_image, model, img_shape=224):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    image = np.asarray(upload_image)
    img = tf.image.resize(image, [img_shape, img_shape])
    img = tf.cast(img, tf.float32)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = food_list[pred_prob.argmax()]
    return pred_class, pred_prob


model, file = setup()
if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions, pred_prob = upload_predict(image, model)
    image_class = str(predictions)
    score = np.round(pred_prob.max() * 100)
    st.write("This is", image_class)
    st.slider("Food Sight🍕👀 Confidence(%)", 0, 100, int(score), disabled=True)

# if img_file_buffer is None:
#     st.text("")
# else:
#     image = Image.open(img_file_buffer)
#     st.image(image, use_column_width=True)
#     predictions, pred_prob = upload_predict(image, model)
#     image_class = str(predictions)
#     score = np.round(pred_prob.max() * 100)
#     st.write("This is", image_class)
#     st.write(f"Food Sight is {score}% confident")
