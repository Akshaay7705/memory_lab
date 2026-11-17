import streamlit as st

st.set_page_config(
    page_title="My Website",
    page_icon="🌐"
)

st.title("Welcome to My Website")

st.write("This is my deployed website.")

name = st.text_input("Enter your name")

if st.button("Submit"):
    st.success(f"Hello, {name}!")