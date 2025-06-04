import streamlit as st
from recommender import load_data, create_similarity_matrix, recommend

st.title("🎬 Movie Recommender")

df = load_data()
similarity = create_similarity_matrix(df)

movie = st.selectbox("Select a Movie:", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(movie, df, similarity)
    st.write("### Recommended Movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
