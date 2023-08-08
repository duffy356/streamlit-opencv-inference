# streamlit image template matching app for object detection based on pyautogui's locateOnScreen with open-cv

This app ships an interactive user interface for template matching in images. 

## Documentation of UI

You can see the selected image with the section of the needle image and the parameters on the left side. The needle image with the hyperparameters confidence and grayscale are applied on the haystack image and the results are shown in the right column. 

The confidence of hits can be changed with the confidence slider. The 'reduce hits' checkbox can be used to reduce overlapping hits to the most confident hits.

## Documentation of Repository

The code for the streamlit user interface is located in Home.py.  
NeedleHaystackSolver.py contains the functions for loading the image and matching the template based on opencv2.   
Examples.py contains code for demonstrating the embedded examples.

## Further readings

Article 1: [Composing an image template matching web app with just 200 LOC](https://medium.com/@dominik.bauer/composing-an-image-template-matching-web-app-with-just-200-loc-13dcf8a77e14)  
Article 2: [Reviewing the locateOnScreen function of PyAutoGUI](https://medium.com/@dominik.bauer/reviewing-the-locateonscreen-function-of-pyautogui-bb82ddf80739)


