import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from numpy import asarray
from rich.box import Box

from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw

from app.service.NeedleHaystackHelper import NeedleHaystackHelper

needle_haystack_helper = NeedleHaystackHelper()

cwd = os.getcwd()

def get_resouce_1path() -> Path:
    return Path(cwd).joinpath("resources")

st.header("Needle Haystack inference")

resource_path = get_resouce_path()
haystack_path = resource_path.joinpath("haystack")

st.markdown("Choose a haystack image from the example images or upload your own!")
col1, col2 = st.columns(2)
selected_image = col1.selectbox("Selected example image", sorted(os.listdir(haystack_path)))
needle_haystack_helper.selected_example_image = selected_image
img_path = haystack_path.joinpath(selected_image).absolute()
image_file = str(img_path)

uploaded_image_file = col2.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
if uploaded_image_file is not None:
    needle_haystack_helper.selected_example_image = None
    image_file = uploaded_image_file

haystack_image = Image.open(image_file)

st.divider()

#st.image(haystack_image)

col3, col4 = st.columns(2)
# Get a cropped image from the frontend
with col3.container():
    cropped_img = st_cropper(haystack_image, box_color="red", box_algorithm=needle_haystack_helper.get_example_boxes, should_resize_image=True)
#cropped_box = st_cropper(haystack_image, box_color="red", return_type='box')
#print(cropped_box)
# Manipulate cropped image at will


    cv_haystack_image = np.array(haystack_image)
#cv2_haystack = cv2.imread(haystack_image)
    cv_cropped_img = np.array(cropped_img)

    cropped_img.thumbnail((150,150))
    col5, col6 = col3.columns([0.2, 0.8], gap="small")
    col5.write("selected Needle image")
    col5.image(cropped_img)
    confidence = col6.slider("confidence", min_value=0.25, max_value=1.00, value=0.95)
    reduce_hits = col6.checkbox("reduce hits", value=True)
    grayscale = col6.checkbox("grayscale", value=True)

with col4.container():
    locations = needle_haystack_helper.locate_all_opencv(cv_cropped_img, cv_haystack_image, confidence=confidence, grayscale=grayscale)
    if locations is None:
        col4.text("no match!")
    modified_image = haystack_image.copy()
    draw = ImageDraw.Draw(modified_image)

    locations_ordered = sorted(list(locations), reverse=True)
    non_overlapping_best_fits = []
    for r in locations_ordered:
        confidence = r[0]
        box = r[1]
        if reduce_hits:
            add = True
            for fit in non_overlapping_best_fits:
                center_box_x = box.left + box.width / 2
                center_box_y = box.top + box.height / 2
                fit_l = fit.left
                fit_r = fit.left + fit.width
                fit_t = fit.top
                fit_b = fit.top + fit.height
                if (fit_l <= center_box_x <= fit_r) and (fit_t <= center_box_y <= fit_b):
                    add = False
                    break
            if add:
                non_overlapping_best_fits.append(box)
        else:
            non_overlapping_best_fits.append(box)

    print(f"draw {len(non_overlapping_best_fits)}")
    for fit in non_overlapping_best_fits:
        draw.rectangle(
            [(fit.left, fit.top), (fit.left + fit.width, fit.top + fit.height)],
            outline='red',
            width=5
        )

    im = asarray(modified_image)
    col4.image(im)
