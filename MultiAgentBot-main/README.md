1. Create Virtual Environment
    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    venv\Scripts\activate

2. pip install -r requirements.txt

3. Download Embedding Model
<!-- from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('path/to/your/model/directory') -->


# Running the Application
1. Activate Virtual Environment

    .venv\Scripts\activate

2. Start the Streamlit App
    streamlit run streamlit_app.py

3. Login Credentials
    Customer Portal

        Email: rohit.sharma@sharmamanufacturing.com
        Password: customer123

        Or:

        Email: priya.patel@techsolutions.in
        Password: customer123

    Underwriter Portal

        Username: admin
        Password: admin123

# Usage Guide
For Underwriters

1.Select "Underwriter" from the role dropdown
2.Login with admin credentials
3.Select insurance type (Health/Life/Motor/Commercial)
4.Upload PDF or TXT documents
5.Click "Analyze" to process documents
6.Ask questions about the documents using the chat interface

For Customers

1.Select "Customer" from the role dropdown
2.Login with customer credentials
3.View your policies
4.Ask questions about coverage, claims, benefits, etc.

# Document Type Separation
The system maintains strict separation between insurance types:

    Health documents are only analyzed when "health" is selected
    Life documents are only analyzed when "life" is selected
    Questions are answered only from documents of the selected type

# ChromaDB Error: 
    Delete storage/chroma folder and restart

# Reset Application
To completely reset the application:
## Delete storage files
rm -rf storage/
rm -rf __pycache__/
rm -rf src/__pycache__/