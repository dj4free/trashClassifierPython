import os
import numpy as np
from PIL import Image
from taipy.gui import Gui
import random
from tensorflow.keras import models

# Define Model
cnn_model = models.load_model('Jupyter/TrashClassifier_best_model.h5')

# Variables
img_path = "elements/placeholder.png"
content = ""
prob = 0
prediction = ""

# Class Names
class_names = {
    0: 'Cardboard',
    1: 'Food Organic',
    2: 'Glass',
    3: 'Metal',
    4: 'Misc',
    5: 'Paper',
    6: 'Plastic',
    7: 'Vegetation',
}


# Define function to predict image input
def predict_img(model, path_to_image):
    """
    Predicts the class of the given image using the provided model.

    Parameters: - model (tensorflow.keras.Model): The pre-trained CNN model. *This is default to best_model from
                TrashClassifier.ipynb* - path_to_image (str): Path to the image file.

    Returns:
    - top_prob (float): The highest probability of the predicted class.
    - class_pred (str): The name of the predicted class.
    """
    img = Image.open(path_to_image)
    img = img.convert('RGB')
    img = img.resize((128, 128))
    data = np.asarray(img)
    data = data / 255.0
    data = np.expand_dims(data, axis=0)
    probs = model.predict(data)
    top_prob = probs.max()
    class_pred = class_names[np.argmax(probs)]

    return top_prob, class_pred


# Define function to pick random image sample on button press
def button_pressed(state):
    """
        Handles 'Random' button action.
        Selects an image at random from the 'sampleImages' directory.
        Updates the image displayed to the randomly chosen image and updates the Prediction state.

        Parameters:
        - state (taipy.gui.State): The state 'Random' button.

        Returns:
        - N/A
        """
    print("Button pressed")
    files = os.listdir("sampleImages")
    images = [file for file in files if file.endswith(('.jpg', '.png'))]
    random_image = random.choice(images)
    state.img_path = os.path.abspath(os.path.join("sampleImages", random_image))
    print(f"Selected random image: {state.img_path}")
    top_prob, top_pred = predict_img(cnn_model, state.img_path)
    state.prob = round(top_prob * 100, 2)
    state.prediction = f"WasteWise is {state.prob}% confident that this is {top_pred} waste."
    state.content = state.img_path  # Make sure content is updated


# Define actions when image is uploaded
def on_change(state, var_name, new_img_path):
    """
        Handles the upload image action.
        Updates the image displayed to the uploaded image and updates the Prediction state.

        Parameters:
        - state (taipy.gui.State): The state file_selector 'upload' button.
        - var_name (str): The name of the variable that triggered the change, defaulted to "content".
        - new_img_path (any): The new path to the image that was uploaded.

        Returns:
        - N/A
        """
    if var_name == "content":
        state.img_path = new_img_path
        top_prob, top_pred = predict_img(cnn_model, new_img_path)
        state.prob = round(top_prob * 100, 2)
        state.prediction = f"WasteWise is {state.prob}% confident that this is {top_pred} waste."


# UI Formatting / Markdown
index = """
<|layout|columns=1|

<|column|
<|text-center|
<|{"elements/WasteWise.png"}|image|width=20vw|>
{: .half-transparent}

|>
|>
|>

<p></p>

<|layout|columns= 1|
<|column|
<|text-center|


## Choose Upload/Random Image
|>
|>
|>

<|layout|columns=1 1|

<|column|
<|text-center|
<|{content}|file_selector|extensions=.jpg|>

*Please upload a photo from your filesystem.*
|>
|>

<|column|
<|text-center|
<|Random|button|on_action=button_pressed|>

*Press for random sample image.*
|>
|>
|>

<p></p><p></p><p></p><p></p>

<|layout|columns=1|width=20vw|
<|text-center|
<|column|
<|{img_path}|image|width=35vw|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=40vw|>

### Prediction:
<|{prediction}|text|>
|>

|>
|>
"""

# StyleKit variables found at https://docs.taipy.io/en/develop/manuals/gui/styling/stylekit/
stylekit = {
    "color_primary": "#4CAF50",  # Primary color
    "color_secondary": "#FFC107",  # Secondary color
    "color_background_light": "#F0F5F7",  # Light background color
    "color_background_dark": "#263238",  # Dark background color
    "color_paper_light": "#FFFFFF",  # Light paper color
    "color_paper_dark": "#37474F",  # Dark paper color
    "border_radius": 10,  # Border radius for rounded corners
    "input_button_height": "50px",  # Height for buttons and inputs
    "font_family": "'Roboto', sans-serif",  # Font family
}

app = Gui(page=index)

if __name__ == '__main__':
    # use_reloader enables automatic reloading
    app.run(debug=False, stylekit=stylekit, title="WasteWise", host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
