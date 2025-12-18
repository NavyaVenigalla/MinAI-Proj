import streamlit as st
from recommender import recommend
import os

# Page configuration
st.set_page_config(
    page_title="Occasion-Based Clothes Recommendation",
    page_icon="👗",
    layout="wide"
)

# Sidebar controls
occasions = ["Home", "Travel", "Office", "Temple", "Function"]

st.sidebar.header("Select Preferences")
occasion = st.sidebar.selectbox("Occasion", occasions)
top_k = st.sidebar.slider("Recommendations", 1, 5, 3)

# Main title
st.title("👗 Occasion-Based Clothes Recommendation System")

st.divider()

# Recommendation action
if st.button("Recommend Clothes"):
    results = recommend(occasion, top_k)

    cols = st.columns(len(results))
    for col, img_path in zip(cols, results):
        with col:
            st.image(img_path, use_container_width=True)
            st.caption(os.path.basename(img_path))

