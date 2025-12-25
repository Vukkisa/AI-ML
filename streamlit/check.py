import streamlit as st
st.header("Streamlit validation Check")
st.write("This is the check module in the streamlit package.")
st.subheader("Functionality")
st.button("Click me!")
sdis=st.text_input("Enter some text:")
#sdis
st.slider("Select a value:", 0, 100, 50)
st.write("If you see this page rendered correctly, the check module is working fine.")
st.number_input("Enter a number:")
file=st.file_uploader("Upload a file to be tested:", type=["txt", "csv", "png", "jpg"])
file.type
if file is not None:
    st.write("File uploaded successfully:", file.name)
    st.image(file)
