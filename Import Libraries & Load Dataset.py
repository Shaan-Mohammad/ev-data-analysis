import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
df = pd.read_excel("C:/Users/cforc/OneDrive/Desktop/LPU classes/K23FK SEM4/INT375 DATA SCIENCE TOOLBOX PYTHON PROGRAMMING/Electric_Vehicle_Population_Data....xlsm")

# Display first few rows
print(df.head())

# Dataset shape and column names
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Check for null values
print("Null values in each column:\n", df.isnull().sum())

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing model_year with most frequent value
df['model_year'] = df['model_year'].fillna(df['model_year'].mode()[0])

# Convert model_year to integer
df['model_year'] = df['model_year'].astype(int)

# Preview cleaned data
print(df.head())

# Summary statistics of numeric columns
print(df.describe())

# Most common EV makes
print("Top 10 makes:\n", df['make'].value_counts().head(10))

# Count of EVs by city
print("Top 10 cities:\n", df['city'].value_counts().head(10))

# Filter: Vehicles with electric range > 200 miles
high_range_vehicles = df[df['electric_range'] > 200]
print("Vehicles with range > 200:\n", high_range_vehicles[['make', 'model', 'electric_range']].head())

# Sort by model year (descending)
sorted_by_year = df.sort_values(by='model_year', ascending=False)
print("Sorted by model year:\n", sorted_by_year[['make', 'model', 'model_year']].head())

# Count of EVs per year
grouped_by_year = df.groupby('model_year').size()
print("EV count by year:\n", grouped_by_year)

# Count of vehicles by make and model
make_model_count = df.groupby(['make', 'model']).size().reset_index(name='count')
print("Top make-model combos:\n", make_model_count.sort_values(by='count', ascending=False).head())

# Define variables needed for plots
top_makes = df['make'].value_counts().head(10)
top_cities = df['city'].value_counts().head(10)
yearly_counts = df['model_year'].value_counts().sort_index()
top_5_makes = df['make'].value_counts().head(5).index
df_top_makes = df[df['make'].isin(top_5_makes)]
ev_type_counts = df['electric_vehicle_type'].value_counts()

# ------------------- Visualizations -------------------

# Top 10 EV Makes
plt.figure(figsize=(10,6))
top_makes.plot(kind='bar', color='dodgerblue', edgecolor='white')
plt.title("Top 10 Electric Vehicle Makes", fontsize=14)
plt.xlabel("Make")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Top 10 Cities with Most EVs
plt.figure(figsize=(10,6))
top_cities.plot(kind='bar', color='darkorange', edgecolor='white')
plt.title("Top 10 Cities with Most Electric Vehicles", fontsize=14)
plt.xlabel("City")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Electric Range Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['electric_range'], bins=30, kde=True, color='mediumpurple', edgecolor='white')
plt.title("Distribution of Electric Range", fontsize=14)
plt.xlabel("Electric Range (Miles)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# EV Adoption Over Years
plt.figure(figsize=(10,6))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', color='limegreen', markersize=6, linewidth=2)
plt.title("EV Adoption Over the Years", fontsize=14)
plt.xlabel("Model Year")
plt.ylabel("Number of Vehicles")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Boxplot: Electric Range by Top 5 Makes
plt.figure(figsize=(10,6))
sns.boxplot(x='make', y='electric_range', data=df_top_makes, 
            hue='make', palette='Set2', linewidth=1.5, legend=False)
plt.title("Electric Range Distribution by Top 5 Makes", fontsize=14)
plt.xlabel("Make")
plt.ylabel("Electric Range (Miles)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Pie Chart: Electric Vehicle Type
plt.figure(figsize=(8,8))
plt.pie(ev_type_counts, labels=ev_type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
plt.title("Electric Vehicle Type Distribution", fontsize=14)
plt.tight_layout()
plt.show()

# Scatter Plot: Electric Range vs. Model Year
plt.figure(figsize=(10,6))
plt.scatter(df['model_year'], df['electric_range'], alpha=0.5, color='teal', edgecolor='white', linewidth=0.3)
plt.title("Electric Range vs. Model Year", fontsize=14)
plt.xlabel("Model Year")
plt.ylabel("Electric Range (Miles)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features Only)", fontsize=14)
plt.tight_layout()
plt.show()

