import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Separate target variable and other variables
    Y = df.Attrition
    X = df.drop(columns=['Attrition'])
    
    # Scale the data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    
    return X_train, X_test, y_train, y_test
