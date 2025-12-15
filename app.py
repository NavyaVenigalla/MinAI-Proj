import streamlit as st
from recommender import recommend
import os

st.set_page_config(page_title="Occasion-Based Clothes Recommendation")

st.title("👗 Occasion-Based Clothes Recommendation System")

# MUST match folder names exactly
occasions = ["Temple", "Function", "Office", "Travel", "Home"]

occasion = st.selectbox("Select Occasion", occasions)
top_k = st.slider("Number of Recommendations", 1, 3, 5)

if st.button("Recommend Clothes"):
    results = recommend(occasion, top_k)

    st.subheader("Recommended Clothes")
    for img_path in results:
        st.image(img_path, width=250)
