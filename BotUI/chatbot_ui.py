import streamlit as st

# Header
st.title("Smart Insurance Assistance")
st.subheader("A Multi- AI Agent powered ChatBot")

# section for files upload options only to specific roles
st.sidebar.title("User Login")
user_role = st.sidebar.selectbox("Select your role:", ["Guest", "Underwriter", "Admin", "Customer", "Sales Agent"])

if user_role in ["Underwriter", "Admin"]:
    # Add a login screen with Username and Password fields
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Submit"):
        if username and password:
            st.sidebar.success(f"Welcome, {username}!")
            st.sidebar.subheader("Upload Files")
            uploaded_files = st.sidebar.file_uploader("Authorized users can upload policy documents or relevant files to the secure database using the Browse files option below", accept_multiple_files=True)
        else:
            st.sidebar.error("Please enter both username and password.")

# Text Area for interacting with the Bot
user_input = st.text_area("max 500 words:", max_chars=500)

# Modify the arrow icon to act as a submit button
st.markdown(
    """
    <style>
    .stTextArea div[data-testid='stMarkdownContainer']::after {
        content: '➤';
        position: absolute;
        right: 10px;
        bottom: 5px;
        font-size: 1.2em;
        cursor: pointer;
    }
    </style>
    <script>
    const textArea = document.querySelector('.stTextArea textarea');
    const arrowIcon = document.querySelector('.stTextArea div[data-testid="stMarkdownContainer"]::after');
    arrowIcon.addEventListener('click', () => {
        textArea.dispatchEvent(new Event('change', { bubbles: true }));
    });
    </script>
    """,
    unsafe_allow_html=True
)

# Handle the submission when the arrow icon is clicked
if user_input:
    st.write(f"You said: {user_input}")

