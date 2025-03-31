import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(
    page_title="Healthcare Data Visualization",
    page_icon="🏥",
    layout="wide"
)

# Add title and description
st.title("Healthcare Data Visualization Dashboard")
st.markdown("""
This dashboard displays key metrics and visualizations for healthcare patient data, 
focusing on readmission rates and length of stay across different departments and treatment protocols.
""")

# Function to generate sample data
@st.cache_data
def generate_sample_data(num_patients=100):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Patient demographic data
    ages = np.random.normal(65, 15, num_patients).astype(int)  # Age centered around 65
    genders = np.random.choice(['Male', 'Female'], num_patients)
    departments = np.random.choice(['Cardiology', 'Neurology', 'Orthopedics', 'Internal Medicine'], num_patients)
    
    # Clinical outcomes
    length_of_stay = np.random.gamma(5, 2, num_patients).astype(int) + 1  # Length of stay in days
    readmitted = np.random.choice([0, 1], num_patients, p=[0.8, 0.2])  # 0: not readmitted, 1: readmitted
    treatment_protocols = np.random.choice(['Protocol A', 'Protocol B', 'Protocol C'], num_patients)
    
    # Create the dataframe
    data = pd.DataFrame({
        'PatientID': range(1, num_patients + 1),
        'Age': ages,
        'Gender': genders,
        'Department': departments,
        'TreatmentProtocol': treatment_protocols,
        'LengthOfStay': length_of_stay,
        'Readmitted': readmitted
    })
    
    return data

# Option to upload own data or use sample data
st.sidebar.header("Data Options")
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
data = None

if use_sample_data:
    num_patients = st.sidebar.slider("Number of Sample Patients", 50, 500, 100)
    data = generate_sample_data(num_patients)
    st.sidebar.success(f"Generated sample data with {num_patients} patients")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your healthcare data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            required_columns = ['PatientID', 'Age', 'Gender', 'Department', 'TreatmentProtocol', 'LengthOfStay', 'Readmitted']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.sidebar.info("Using sample data instead")
                data = generate_sample_data()
            else:
                st.sidebar.success("Data successfully loaded")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            st.sidebar.info("Using sample data instead")
            data = generate_sample_data()
    else:
        st.sidebar.info("Please upload a CSV file or use sample data")
        data = generate_sample_data()

# Display data summary
with st.expander("View Dataset"):
    st.dataframe(data)
    
    st.subheader("Dataset Summary")
    st.write(data.describe(include='all'))

# Key metrics
st.header("Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Length of Stay", f"{data['LengthOfStay'].mean():.2f} days")
    
with col2:
    st.metric("Readmission Rate", f"{data['Readmitted'].mean() * 100:.2f}%")
    
with col3:
    st.metric("Total Patients", f"{len(data)}")
    
with col4:
    st.metric("Departments", f"{data['Department'].nunique()}")

# Create visualizations
st.header("Visualizations")

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Patient Ages")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='Age', kde=True, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)

with col2:
    st.subheader("Readmission Rates by Treatment Protocol")
    fig, ax = plt.subplots(figsize=(10, 6))
    readmission_by_protocol = data.groupby('TreatmentProtocol')['Readmitted'].mean() * 100
    readmission_by_protocol.plot(kind='bar', color=sns.color_palette("Set2"), ax=ax)
    ax.set_xlabel('Treatment Protocol')
    ax.set_ylabel('Readmission Rate (%)')
    st.pyplot(fig)

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Length of Stay by Department")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Department', y='LengthOfStay', data=data, ax=ax)
    ax.set_xlabel('Department')
    ax.set_ylabel('Length of Stay (Days)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("Correlation Between Age and Length of Stay")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Age', y='LengthOfStay', hue='Readmitted', 
                    palette={0: 'blue', 1: 'red'}, alpha=0.7, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Length of Stay (Days)')
    ax.legend(title='Readmitted', labels=['No', 'Yes'])
    st.pyplot(fig)

# Row 3 - Full width
st.subheader("Correlation Matrix of Patient Data")
# Convert categorical variables to numeric for correlation
data_numeric = pd.get_dummies(data.drop('PatientID', axis=1), drop_first=True)
correlation_matrix = data_numeric.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Row 4 - Department & Gender Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Distribution by Department")
    dept_counts = data['Department'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(dept_counts, labels=dept_counts.index, autopct='%1.1f%%', 
          colors=sns.color_palette("pastel"), startangle=90)
    st.pyplot(fig)

with col2:
    st.subheader("Gender Distribution")
    gender_counts = data['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(gender_counts.index, gender_counts, color=sns.color_palette("pastel"))
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Protocol Performance Analysis
st.header("Protocol Performance Analysis")
protocol_analysis = data.groupby('TreatmentProtocol').agg({
    'Readmitted': ['mean', 'count'],
    'LengthOfStay': 'mean'
}).reset_index()
protocol_analysis.columns = ['Protocol', 'ReadmissionRate', 'PatientCount', 'AvgLOS']
protocol_analysis['ReadmissionRate'] = protocol_analysis['ReadmissionRate'] * 100

st.dataframe(protocol_analysis)

# Add a conclusion
st.header("Analysis Summary")
st.markdown("""
Based on the visualizations above, we can observe:
- The readmission rates vary significantly across different treatment protocols
- There appears to be a correlation between patient age and length of stay
- Some departments show higher average length of stay than others
- Protocol performance analysis provides insights into which protocols might need review

These insights can help healthcare providers make data-driven decisions to improve patient outcomes and operational efficiency.
""")

# Add footer
st.markdown("---")
st.markdown("Healthcare Data Analytics Dashboard | Created with Streamlit")