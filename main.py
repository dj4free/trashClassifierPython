import os
import numpy as np
from PIL import Image
from taipy.gui import Gui
import random
from tensorflow.keras import models

os.environ["PATH"] += os.pathsep + 'C:\\Users\\djcum\\Graphviz-11.0.0-win64\\bin'

# Define Model
cnn_model = models.load_model('Jupyter/TrashClassifier.h5')

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
    img = Image.open(path_to_image)
    img = img.convert('RGB')
    img = img.resize((128, 128))  # Ensure the image is resized to 128x128
    data = np.asarray(img)
    data = data / 255.0
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    probs = model.predict(data)
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]

    return top_prob, top_pred


# Define function to pick random image sample on button press
def button_pressed(state):
    print("Button pressed")
    files = os.listdir("sampleImages")
    images = [file for file in files if file.endswith(('.jpg', '.png'))]
    random_image = random.choice(images)
    state.img_path = os.path.abspath(os.path.join("sampleImages", random_image))
    print(f"Selected random image: {state.img_path}")
    top_prob, top_pred = predict_img(cnn_model, state.img_path)
    state.prob = round(top_prob * 100, 2)
    state.prediction = f"WasteWise is {state.prob}% confident that this is {top_pred} waste."
    state.content = state.img_path  # Ensure content is updated


# Define actions when image is uploaded
def on_change(state, var_name, var_value):
    if var_name == "content":
        state.img_path = var_value
        top_prob, top_pred = predict_img(cnn_model, var_value)
        state.prob = round(top_prob * 100, 2)
        state.prediction = f"WasteWise is {state.prob}% confident that this is {top_pred} waste."


# on_init function
def on_init(state):
    try:
        print("Initializing application...")
        absolute_path = os.path.abspath("elements/placeholder.png")
        print(f"Resolved absolute path: {absolute_path}")
        if os.path.exists(absolute_path):
            state.img_path = absolute_path
            state.content = absolute_path  # Initialize content with the placeholder image
            print(f"Initial img_path set to: {state.img_path}")
        else:
            print(f"File not found: {absolute_path}")
        state.prob = 0
        state.prediction = "Please upload an image to get started."
    except Exception as e:
        print(f"Error in on_init: {e}")


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
