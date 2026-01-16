# Minor in AI Project(Content Recommendation System)
This project is an AI-powered "Occasion-Based Clothes Recommendation System" that suggests suitable clothing based on the selected occasion such as Temple, Function, Office, Travel, and Home.
The system uses image–text similarity with the CLIP model to recommend visually relevant outfits from a curated dataset of clothing images. Users can easily select an occasion through an interactive Streamlit-based web interface, making the experience simple and user-friendly.

Problem Statement:
Choosing the right outfit for different occasions can be challenging due to the wide variety of clothing styles, colors, and seasonal trends. Users often struggle to decide what to wear for a specific event. This system addresses the problem by providing personalized clothing recommendations based on occasion-specific requirements.

Tools and Technologies used:
Programming Language: Python
Framework: Streamlit
Machine Learning Technique: Content-Based Recommendation System, The recommendation logic is based on content similarity.
Libraries Used: NumPy, Pandas, Scikit-learn, CLIP
Development Environment: VS Code

System Design:
1.User selects an occasion from the UI.
2.The system loads clothing images corresponding to the selected occasion.
3.A descriptive text prompt is generated for the selected occasion.
4.The CLIP model encodes both images and text into a shared embedding space.
5.Cosine similarity is calculated between image and text embeddings.
6.The top-matching clothing images are displayed to the user.

Results:
Accurate clothing recommendations for each selected occasion
No cross-mixing of clothing categories
Improved recommendation quality using descriptive text prompts
Real-time image ranking based on semantic similarity

Key Learnings:
Understanding multimodal machine learning models
Implementing content-based recommendation systems
Using CLIP for image–text similarity
Importance of prompt engineering
Integrating ML models with Streamlit UI
