import streamlit as st

# Header
st.title("Smart Insurance Assistance")
st.subheader("A Multi- AI Agent powered ChatBot")

# section for files upload options only to specific roles
st.sidebar.title("Upload Files")
user_role = st.sidebar.selectbox("Select your role:", ["Underwriter", "Admin", "Customer", "Sales Agent"])

if user_role in ["Underwriter", "Admin"]:
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file:
        st.sidebar.write(f"Uploaded file: {uploaded_file.name}")

# Text Area for interacting with the Bot
user_input = st.text_area("max 500 words:", max_chars=500)

# Submit Button
if st.button("Submit"):
    if user_input:
        st.write(f"You said: {user_input}")
    else:
        st.write("Please enter some text to interact with the bot.")
