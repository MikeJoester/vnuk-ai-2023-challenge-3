from nltk.stem.porter import PorterStemmer
import time
from nltk.corpus import stopwords
import streamlit as st
import pickle
import string
import re
import nltk
import tensorflow as tf
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

# hide_menu = """
# <style>
# #MainMenu{
#     visibility:hidden;
# }
# footer{
#     visibility:hidden;
# }
# </style>
# """
showWarningOnDirectExecution = False
ps = PorterStemmer()

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]
{
    background-image: url("https://e0.pxfuel.com/wallpapers/259/322/desktop-wallpaper-technology-ultra-technology.jpg");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# st.set_page_config(page_title="Cyberbullying Detection", page_icon=image)

# st.markdown(hide_menu, unsafe_allow_html=True)

# st.sidebar.markdown("<br>", unsafe_allow_html=True)
# st.sidebar.image(image, use_column_width=True, output_format='auto')


# st.sidebar.markdown("---")


# st.sidebar.markdown(
#     "<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | baodeptrai</h1>", unsafe_allow_html=True)


def clean_text(tweet):
    # remove URL
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove usernames
    tweet = re.sub(r'@[^\s]+[\s]?', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#[^\s]+[\s]?', '', tweet)
    # Remove emojis
    tweet = re.sub(r':[^\s]+[\s]?', '', tweet)
    # remove special characters
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)
    # remove RT
    tweet = re.sub('RT', '', tweet)
    # remove Numbers
    tweet = re.sub('[0-9]', '', tweet)

    return tweet


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


model = tf.keras.models.load_model('my_model.h5')
model.make_predict_function()

with open('vectorization_config.pkl', 'rb') as f:
    tfidf = pickle.load(f)
loaded_vectorization_layer = tf.keras.layers.TextVectorization.from_config(
    tfidf['config'])
loaded_vectorization_layer.set_vocabulary(tfidf['vocabulary'])
st.title("Vietnamese Profanity Checker")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
input_text = st.text_area("**_Input text in the box below!!!_**",
                          key="**_Enter the text to analyze_**")
col1, col2 = st.columns([1, 6])
with col1:
    button_predict = st.button('Check')
with col2:

    def clear_text():
        st.session_state["**_Enter the text to analyze_**"] = ""

    # clear button
    button_clear = st.button("Clear", on_click=clear_text)

st.markdown("---")
# predict button animations
if button_predict:
    if input_text == "":
        st.snow()
        st.warning("No text found, please try again!")
    else:
        with st.spinner("**_Checking_** Please wait..."):
            time.sleep(3)
    # 1. preprocess

        cleanText = clean_text(input_text)

        transformText = transform_text(cleanText)

    # 2. vectorize

        vectorized_input = loaded_vectorization_layer([input_text])
    # 3. predict

        result = model.predict(vectorized_input)[0]
        result = np.argmax(result)
        # result2 = model.predict_proba(vector_input)[0]
        # clf=svm.SVC(probability=True)

    # 4. display
        if result == 2:
            st.subheader("The given text seems to be:")
            st.warning(":yellow[**Slightly profanity**]")
            # st.markdown(result2)
        elif result == 1:
            st.subheader("The given text seems to be:")
            st.error(":red[**Definitely profanity**]")
            # st.markdown(result2)
        else:
            st.subheader("The given text seems to be:")
            st.success(":green[**Normal**]")