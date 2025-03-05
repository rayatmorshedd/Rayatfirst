import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and display data
def load_and_display_data(file_path):
    data = pd.read_csv(file_path)
    print(data.info())
    print("\nDataset Overview:")
    print(f"\nDataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    print("\nFirst few rows of the dataset:")
    print(data.head())
    return data


# Convertion of date column
def convert_date(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Verify conversion success
        print("\nDate column successfully converted. Sample values:")
        print(df[['Date', 'Year', 'Month']].head())
    return df



# Handle the missing values

def clean_null_values(dataset):
    print("\nMissing values in each column before cleaning:")
    print(dataset.isnull().sum()[dataset.isnull().sum() > 0])

    if 'Age' in dataset.columns:
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    for col in ['Sex', 'Race', 'Ethnicity', 'Manner of Death']:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    # Verify missing values after cleaning
    print("\nMissing values after cleaning:")
    print(dataset.isnull().sum()[dataset.isnull().sum() > 0])

    return dataset  # Return the modified dataset


# Dropping any unnecessary columns
def  drop_columns(data, columns):
    data = data.copy()
    data.drop(columns=[col for col in columns if col in data.columns], inplace=True)
    return data


# Display the basic statistics
def display__statistics(df):
    print("\nBasic Statistics:")
    print(df.describe())


# Encoding categorical variables
def convert_categorical_data(data):
    for col in ['Race', 'Sex', 'Ethnicity', 'Manner of Death']:
        if col in data.columns:
            data[col] = data[col].astype(str)  #
    return pd.get_dummies(data, drop_first=True)

# Death count through year and gender
def counting_deaths(df):
    yearly_deaths = df['Year'].value_counts().sort_index() if 'Year' in df.columns else None
    gender_deaths = df['Sex_Male'].value_counts() if 'Sex_Male' in df.columns else None
    return yearly_deaths, gender_deaths


# Death count by race
def count_deaths_by_race(data):
    race_columns = [col for col in data.columns if col.startswith('Race_')]
    race_counts = data[race_columns].sum() if race_columns else None
    return race_counts


def frequent_drug_descriptions(df):
    if 'Description of Injury' in df.columns:
        print("\nTop 10 Most Frequent Drug Descriptions:")
        print(df['Description of Injury'].value_counts().head(10))


# Visualization of function
def visualize_data(yearly_deaths, gender_deaths, race_counts, df):

    if yearly_deaths is not None:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_deaths, marker="o", color="blue")
        plt.title('Trend of Drug-Related Deaths Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Deaths')
        plt.grid()
        plt.show()

    if race_counts is not None:
        plt.figure(figsize=(12, 6))
        race_counts.sort_values(ascending=False).plot(kind='bar', color='purple', alpha=0.7)
        plt.xlabel("Race")
        plt.ylabel("Count")
        plt.title("Number of Deaths by Race")
        plt.xticks(rotation=45)
        plt.show()

    if gender_deaths is not None:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df['Sex_Male'], hue=df['Sex_Male'], palette='viridis', legend=False)
        plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
        plt.title('Number of Deaths by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.show()

    if 'Sex_Male' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df['Sex_Male'], y=df['Age'])
        plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
        plt.title('Age Distribution by Gender')
        plt.show()

    if 'Age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'], bins=20, kde=True, color='blue', alpha=0.5)
        plt.xticks(range(0, 101, 10))
        plt.xlabel("Age")
        plt.ylabel("Death Count")
        plt.title("Age Distribution of Deaths")
        plt.show()

    drug_columns = [col for col in df.columns if col in ['Cocaine', 'Heroin' , 'Fentanyl', 'Methadone', 'Oxycodone']]
    if drug_columns:
        drug_counts = df[drug_columns].notnull().sum()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=drug_counts.index, y=drug_counts.values, palette='viridis')
        plt.title('Frequency of Drug Presence in Cases')
        plt.xlabel('Drug Type')
        plt.ylabel('Count')
        plt.show()
        drug_data = df[drug_columns].notnull().astype(int)
        plt.figure(figsize=(8, 6))
        sns.heatmap(drug_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Drug Presence')
        plt.show()

    if 'Age' in df.columns:
        age_bins = [0, 18, 30, 40, 50, 60, 100]
        age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '60+']
        df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Age Group', hue='Age Group', palette="coolwarm", legend=False)
        plt.title('Drug-Related Deaths by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.show()

# Main function
def main():
    file_path = "Accidental_Drug_Related_Deaths.csv"
    df = load_and_display_data(file_path)
    df = convert_date(df)
    df = clean_null_values(df)
    df = convert_categorical_data(df)
    df = drop_columns(df, ['ResidenceCityGeo', 'InjuryCityGeo', 'DeathCityGeo'])

    display__statistics(df)

    yearly_deaths, gender_deaths = counting_deaths(df)
    race_counts = count_deaths_by_race(df)

    frequent_drug_descriptions(df)
    visualize_data(yearly_deaths, gender_deaths, race_counts, df)

    df.to_csv('Cleaned_Accidental_Deaths.csv', index=False)
    print("\n Analysis complete. Cleaned data saved.")

if __name__ == "__main__":
    main()
