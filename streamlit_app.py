import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our visualizations
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Create sample healthcare data based on the project example
# In a real scenario, you would load this from a CSV file or database
np.random.seed(42)  # For reproducibility

# Generate sample data for 100 patients
num_patients = 100

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

# Print the first few rows of our dataset
print("Sample Healthcare Dataset:")
print(data.head())
print("\nDataset Summary:")
print(data.describe(include='all').to_string())

# Create visualizations

# 1. Distribution of patient ages
plt.figure()
sns.histplot(data=data, x='Age', kde=True)
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('patient_age_distribution.png')

# 2. Readmission rates by treatment protocol
plt.figure()
readmission_by_protocol = data.groupby('TreatmentProtocol')['Readmitted'].mean() * 100
readmission_by_protocol.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('30-Day Readmission Rate by Treatment Protocol')
plt.xlabel('Treatment Protocol')
plt.ylabel('Readmission Rate (%)')
plt.tight_layout()
plt.savefig('readmission_by_protocol.png')

# 3. Length of stay by department
plt.figure()
sns.boxplot(x='Department', y='LengthOfStay', data=data)
plt.title('Length of Stay by Department')
plt.xlabel('Department')
plt.ylabel('Length of Stay (Days)')
plt.tight_layout()
plt.savefig('los_by_department.png')

# 4. Correlation between age and length of stay
plt.figure()
sns.scatterplot(data=data, x='Age', y='LengthOfStay', hue='Readmitted', 
                palette={0: 'blue', 1: 'red'}, alpha=0.7)
plt.title('Correlation Between Age and Length of Stay')
plt.xlabel('Age')
plt.ylabel('Length of Stay (Days)')
plt.legend(title='Readmitted', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('age_vs_los.png')

# 5. Heatmap showing correlations
plt.figure(figsize=(10, 8))
# Convert categorical variables to numeric for correlation
data_numeric = pd.get_dummies(data.drop('PatientID', axis=1), drop_first=True)
correlation_matrix = data_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Patient Data')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# 6. Dashboard-style summary visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 6.1 Patient distribution by department
dept_counts = data['Department'].value_counts()
axes[0, 0].pie(dept_counts, labels=dept_counts.index, autopct='%1.1f%%', 
              colors=sns.color_palette("pastel"), startangle=90)
axes[0, 0].set_title('Patient Distribution by Department')

# 6.2 Gender distribution
gender_counts = data['Gender'].value_counts()
axes[0, 1].bar(gender_counts.index, gender_counts, color=sns.color_palette("pastel"))
axes[0, 1].set_title('Gender Distribution')
axes[0, 1].set_ylabel('Count')

# 6.3 Average LOS by treatment protocol
avg_los = data.groupby('TreatmentProtocol')['LengthOfStay'].mean().sort_values(ascending=False)
axes[1, 0].bar(avg_los.index, avg_los, color=sns.color_palette("deep"))
axes[1, 0].set_title('Average Length of Stay by Protocol')
axes[1, 0].set_ylabel('Average Days')

# 6.4 Readmission counts
readmit_status = ['Not Readmitted', 'Readmitted']
readmit_counts = [sum(data['Readmitted'] == 0), sum(data['Readmitted'] == 1)]
axes[1, 1].bar(readmit_status, readmit_counts, color=['green', 'red'])
axes[1, 1].set_title('Readmission Status')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('healthcare_dashboard.png')

print("\nAll visualizations have been created and saved.")
print("Generated files:")
print("- patient_age_distribution.png")
print("- readmission_by_protocol.png")
print("- los_by_department.png")
print("- age_vs_los.png")
print("- correlation_heatmap.png")
print("- healthcare_dashboard.png")

# Additional analysis: Calculate key metrics mentioned in the example
print("\nKey Performance Metrics:")
print(f"Average Length of Stay: {data['LengthOfStay'].mean():.2f} days")
print(f"Readmission Rate: {data['Readmitted'].mean() * 100:.2f}%")

# Example of finding correlations that might lead to the insights mentioned
# Group by treatment protocol and calculate readmission rates
protocol_analysis = data.groupby('TreatmentProtocol').agg({
    'Readmitted': ['mean', 'count'],
    'LengthOfStay': 'mean'
}).reset_index()
protocol_analysis.columns = ['Protocol', 'ReadmissionRate', 'PatientCount', 'AvgLOS']
protocol_analysis['ReadmissionRate'] = protocol_analysis['ReadmissionRate'] * 100
print("\nProtocol Performance Analysis:")
print(protocol_analysis.to_string(index=False))