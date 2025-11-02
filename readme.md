# Smart SMS Insights Dashboard

## How to Run the Code

1. Open your project folder (where this script is saved).

2. Ensure the following model files and directories exist:
   - `models/fine_tuned_embedding/`
   - `models/classifier_model.pt`
   - `models/sms_summary_model.pkl`

3. Install the required dependencies:
   ```bash
   pip install streamlit torch sentence-transformers pandas scikit-learn joblib
Run the Streamlit app:

bash
Copy code
streamlit run app.py
After running, Streamlit will display a message like:

nginx
Copy code
Local URL: http://localhost:8501
Open this link in your browser to launch the Smart SMS Insights Dashboard.

How to Use
Select a theme (Light or Dark) from the sidebar.

Choose an input mode:

Upload CSV File: Upload a CSV containing a column named "Message" or similar.

Write / Paste Messages: Enter one message per line in the format Sender: Message.

Click Analyze Messages to process and summarize your data.

The dashboard will display:

A category distribution chart.

Summaries for each sender or message cluster.

A detailed message table.

Use the Download Categorized Messages button to export the analyzed data as a CSV file.
