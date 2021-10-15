import os
import time

# core packages
import streamlit as st
st.set_page_config(page_title="Covid19 Detection Tool", page_icon="covid19.jpeg", layout='centered', initial_sidebar_state='auto')

# visualization packages
import cv2
from PIL import Image,ImageEnhance
import numpy as np 

# NN packages
import tensorflow as tf


# use relative path
#path = os.path.dirname(__file__)


def main():
    """
    Example webapp tool for Covid-19 detection
    """
    title_templ = """
    <div class="container py-5 text-center">
        <h1 class="display-4 font-weight-bold">Covid-19 Detection Demo Tool</h1>
        <p class="font-italic mb-0">A WebApp for Covid-19 Diagnosis powered by CNN Image Classification and Streamlit</p>
    </div>
	"""
    st.markdown(title_templ,unsafe_allow_html=True)

    #st.sidebar.image(path+"/covid19.jpeg", use_column_width=True)
    st.sidebar.image("covid19.jpeg", use_column_width=True)

    # if sample file is used
    checkbox_sample = st.sidebar.checkbox("Use sample file")
    if checkbox_sample:
        image_file = "sample-normal-xray.jpg"
    else:
        image_file = st.sidebar.file_uploader("Upload an X-Ray Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image, use_column_width=True)
        activities = ["Image Enhancement","Diagnosis", "Disclaimer and Info"]
        choice = st.sidebar.selectbox("Select Activity",activities)
        # IMAGE ENHANCEMENT 
        if choice == 'Image Enhancement':
            st.subheader("Image Enhancement")
            enhance_type = st.sidebar.radio("Enhance Type",["Original","Contrast","Brightness"])
            if enhance_type == "Contrast":
                c_rate = st.slider("Contrast",0.5,5.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output,use_column_width=True)
            elif enhance_type == "Brightness":
                c_rate = st.slider("Brightness",0.5,5.0)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output,use_column_width=True)
            else:
                st.text("Original Image")                
                st.image(our_image,use_column_width=True)
            
        # DIAGNOSIS
        elif choice == 'Diagnosis':
            if st.sidebar.button("Diagnosis"):
                # image to black and white
                # our image is binary, we have to convert it to array
                new_img = np.array(our_image.convert('RGB'))
                # 0 is original, 1 is grayscale
                new_img = cv2.cvtColor(new_img,1)
                gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
                st.text("Chest X-RAY")
                st.image(gray,use_column_width=True)
                # X-RAY (Image) Preprocessing
                IMG_SIZE = (200,200)
                img = cv2.equalizeHist(gray)
                img = cv2.resize(img,IMG_SIZE)
                # normalization
                img = img/255.
                # image reshape according to TensorFlow format
                X_Ray = img.reshape(1,200,200,1)
                # load pre-trained CNN model
                model = tf.keras.models.load_model("./models/Covid19_CNN_Classifier.h5")
                # diagnosis (binary classification)
                diagnosis = model.predict_classes(X_Ray)
                diagnosis_proba = model.predict(X_Ray)
                probability_cov = diagnosis_proba*100
                probability_no_cov = (1-diagnosis_proba)*100
                
                my_bar = st.sidebar.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.05)
                    my_bar.progress(percent_complete + 1)
                
                # diagnosis cases: 1=covid, 0=no covid
                if diagnosis == 0:
                    st.sidebar.success("DIAGNOSIS: NO COVID-19 (Probability: %.2f%%)" % (probability_no_cov))
                else:
                    st.sidebar.error("DIAGNOSIS: COVID-19 (Probability: %.2f%%)" % (probability_cov))
                st.warning("Keep in mind that the COVID-19 detection demo tool is for educational purposes only (refer to my “Disclaimer and Info” in the Select Activity dropdown)")


        # DISCLAIMER
        else:
            st.subheader("Disclaimer and Info")
            st.subheader("Disclaimer")
            st.write("**This tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis**")
            st.subheader("Background")
            st.write("This tool gets inspiration from the following works:")
            st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
            st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)") 
            st.write("I used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a Convolutional Neural Network (consisting of 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.")
            st.write("Since the dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since I got a 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.")
            st.write("Unfortunately in the test, I got 5 cases of 'False Negative' patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")
            st.write("The model has the following limitations:")
            st.write("- small dataset (a bigger dataset will most likely help in improving performance)")
            st.write("- images coming only from the PA position")
            st.write("- a fine tuning activity is strongly suggested")
            st.write("")

    # ABOUT THE AUTHOR
    if st.sidebar.button("About the Author"):
        st.sidebar.subheader("Covid-19 Detection Tool")
        st.sidebar.markdown("by [Benjamin Tabares Jr](https://www.linkedin.com/in/benjamin-tabares/)")
        st.sidebar.markdown("[bateou@yahoo.com](mailto:bateou@yahoo.com)")
        st.sidebar.text("All Rights Reserved (2021)")


if __name__ == '__main__':
		main()	