import os
import numpy as np
import streamlit as st
import seaborn as sns
from numpy import asarray

from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw

from service.Examples import Examples
from service.NeedleHaystackSolver import NeedleHaystackSolver

examples = Examples()

st.set_page_config(layout="wide")

st.header("Example Template Matching inference App built with Streamlit!")
st.markdown("Choose a haystack image from the example images or upload your own!")

haystack_path = Examples.get_resouce_path()

# example or own image
col1, col2 = st.columns(2)
selected_image = col1.selectbox("Selected example image", sorted([f for f in os.listdir(haystack_path) if not f.startswith('.')]))
examples.selected_example_image = selected_image
img_path = haystack_path.joinpath(selected_image).absolute()
image_file = str(img_path)

uploaded_image_file = col2.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
if uploaded_image_file is not None:
    examples.selected_example_image = None
    image_file = uploaded_image_file

haystack_image = Image.open(image_file)

# description / docs
st.markdown("You can see the selected image with the section of the needle image and the parameters on the left side. The needle image with the hyperparameters confidence and grayscale are applied on the haystack image and the results are shown in the right column.")
st.markdown("The confidence of hits can be changed with the confidence slider. The 'reduce hits' checkbox can be used to reduce overlapping hits to the most confident hits.")

st.divider()

# example image and cropped image
col3, col4 = st.columns(2)

# Get a cropped image from the frontend for the needle image on the left side
col3.markdown("**Selected Haystack image**")
with col3.container():

    cropped_img = st_cropper(haystack_image, box_color="red", box_algorithm=examples.get_example_boxes, should_resize_image=True)
    cv_haystack_image = np.array(haystack_image)
    cv_cropped_img = np.array(cropped_img)

    cropped_img.thumbnail((150, 150))
    col5, col6 = col3.columns([0.2, 0.8], gap="small")
    col5.markdown("**selected Needle image**")
    col5.image(cropped_img)

    # other options
    confidence = col6.slider("confidence", min_value=0.25, max_value=1.00, value=0.95)
    grayscale = col6.checkbox("grayscale", value=True)
    reduce_hits = col6.checkbox("reduce hits", value=True)

# search and show results on right side
col4.markdown("**Result of Solver**")
with col4.container():
    locations = NeedleHaystackSolver.locate_all_opencv(cv_cropped_img, cv_haystack_image, confidence=confidence, grayscale=grayscale)
    if locations is None:
        col4.text("no match!")
    modified_image = haystack_image.copy()
    result_image = ImageDraw.Draw(modified_image)

    locations_ordered = sorted(list(locations), reverse=True)
    non_overlapping_best_fits = []
    for loc in locations_ordered:
        confidence = loc.confidence
        box = loc.box
        if reduce_hits:
            add = True
            for res in non_overlapping_best_fits:
                center_box_x = box.left + box.width / 2
                center_box_y = box.top + box.height / 2

                fit_box = res.box
                l = fit_box.left
                r = fit_box.left + fit_box.width
                t = fit_box.top
                b = fit_box.top + fit_box.height
                if (l <= center_box_x <= r) and (t <= center_box_y <= b):
                    add = False
                    break
            if add:
                non_overlapping_best_fits.append(loc)
        else:
            non_overlapping_best_fits.append(loc)

    hex_palette = sns.color_palette("Greens", 100).as_hex()

    # draw the hits onto the image
    shown_hits = {}
    for fit in non_overlapping_best_fits:
        confidence = round(fit.confidence * 100)
        confidence_color = hex_palette[confidence - 1]
        box = fit.box

        result_image.rectangle(
            [(box.left, box.top), (box.left + box.width, box.top + box.height)],
            outline=confidence_color,
            width=10
        )
        shown_hits[confidence] = confidence_color

    im = asarray(modified_image)
    col4.image(im, width=Examples.resized_width(haystack_image))

    # show inferences
    col4.markdown("**Confidence by color**")
    for confidence, color in shown_hits.items():
        col4.markdown(f"<span style='color:{color}'>{confidence}</span>",
                    unsafe_allow_html=True)

