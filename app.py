import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Load the datasets
try:
    disease_data = pd.read_excel("updated_dataset_with_remedies.xlsx")
    remedies_data = pd.read_excel("symptoms_remedies.xlsx")
except FileNotFoundError:
    st.error("Dataset files not found. Please upload the datasets in the same directory.")
    st.stop()

# Data preprocessing
# Ensure the 'Disease' column exists in the dataset
if "Disease" not in disease_data.columns:
    st.error("The 'Disease' column is missing in the dataset. Please check the data.")
    st.stop()

# Encode the target variable
label_encoder = LabelEncoder()
disease_data["Disease"] = label_encoder.fit_transform(disease_data["Disease"])

# Separate features (X) and target (y)
X = disease_data.drop(columns=["Disease"], axis=1)
y = disease_data["Disease"]

# Ensure X is numeric
X = X.apply(pd.to_numeric, errors="coerce")
if X.isnull().any().any():
    X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
rf_model = RandomForestClassifier().fit(X_train, y_train)
svm_model = SVC(probability=True).fit(X_train, y_train)
log_reg_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Streamlit UI
st.title("Disease Prediction System")
st.write("Select symptoms and predict the disease with home remedies.")

# Symptom selection
symptoms = X.columns.tolist()
selected_symptoms = st.multiselect("Select Symptoms:", symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create input vector for prediction
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        input_df = pd.DataFrame([input_vector], columns=symptoms)

        # Predict with each model
        rf_pred = rf_model.predict(input_df)[0]
        svm_pred = svm_model.predict(input_df)[0]
        log_reg_pred = log_reg_model.predict(input_df)[0]

        # Majority voting
        predictions = [rf_pred, svm_pred, log_reg_pred]
        final_prediction = max(set(predictions), key=predictions.count)
        predicted_disease = label_encoder.inverse_transform([final_prediction])[0]

        # Display the predicted disease
        st.success(f"Predicted Disease: {predicted_disease}")
        remedy = disease_data.loc[disease_data["Disease"] == final_prediction, disease_data["Remedies"]].values
        if remedy.size > 0:
            st.write("### Home Remedy for Predicted Disease:")
            st.write(remedy[0])
        else:
            st.warning("No remedy found for the predicted disease.")

        # Retrieve home remedies for the predicted disease
     

        # Retrieve remedies for symptoms using cosine similarity
        symptom_remedies = remedies_data["Symptoms"].dropna().tolist()
        vectorizer = CountVectorizer().fit_transform(symptom_remedies)
        vectors = vectorizer.toarray()

        selected_symptom_str = ", ".join(selected_symptoms)
        input_vector = vectorizer.transform([selected_symptom_str]).toarray()
        similarities = cosine_similarity(input_vector, vectors)
        most_similar_idx = similarities[0].argsort()[-5:][::-1]  # Top 5 similar symptoms

        st.write("### Additional Remedies for Selected Symptoms:")
        for idx in most_similar_idx:
            st.write(remedies_data.iloc[idx]["Remedies"])  

