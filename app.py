import streamlit as st
import tensorflow as tf
from PIL import Image
from class_names import food_list
import numpy as np


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("./my_food_sight_model.h5")
    return model


with st.spinner("Model is being loaded.."):
    model = load_model()
st.write(
    """
         # Food Sight
         """
)

file = st.file_uploader(
    "Upload the image to be classified", type=["jpg", "png", "jpeg"]
)
st.set_option("deprecation.showfileUploaderEncoding", False)


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


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions, pred_prob = upload_predict(image, model)
    image_class = str(predictions)
    score = f"{pred_prob.max():.2f}"
    st.write("The image is classified as", image_class)
    st.write("The similarity score is approximately", score)
    print(
        "The image is classified as ", image_class, "with a similarity score of", score
    )
