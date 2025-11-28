🏡 House Price Prediction App

A Streamlit Machine Learning Web App using the California Housing Dataset

This project is a simple but powerful machine learning web application built with Streamlit, Scikit‑Learn, Pandas, and NumPy.

The app predicts house prices in California using user‑entered housing and demographic features. It uses a Linear Regression model trained on the California Housing Dataset, a built‑in dataset from Scikit‑Learn.

⸻

🚀 Live Demo

Streamlit Cloud, link:

https://house-price-predictor-hcbiz9n4mvkiwfynjp26cv.streamlit.app/



⸻

📌 Features
	•	✔ Clean, interactive Streamlit UI
	•	✔ Real‑time house price prediction
	•	✔ Model trained on the built‑in California Housing Dataset
	•	✔ Sliders and number inputs for user‑friendly data entry
	•	✔ Automatic preprocessing + prediction

⸻

📂 Project Structure

📁 house-price-predictor
│── housePrice.py              # Main Streamlit application
│── requirements.txt    # Dependencies needed to run the app
│── README.md           # Project documentation


⸻

🧠 How It Works

1. Dataset

The app uses fetch_california_housing() from Scikit‑Learn.
This dataset contains:
	•	Median income
	•	Average rooms
	•	Average bedrooms
	•	House age
	•	Population
	•	Occupancy
	•	Latitude
	•	Longitude

These features are used to predict Median House Value.

2. Model Training

A Linear Regression model is trained each time the app runs:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

3. User Input

The user provides housing characteristics via Streamlit:
	•	Median income
	•	House age
	•	Average rooms
	•	Bedrooms
	•	Population
	•	Occupancy
	•	Latitude & Longitude

4. Prediction

The model predicts a house price and multiplies it by 100,000 (because the dataset stores price units in $100k):

prediction = model.predict(input_data)[0]
st.write(f"${prediction * 100000:.2f}")


⸻

🛠 Installation & Running Locally

1. Clone the repository

git clone https://github.com/acehunter1307/house-price-predictor
cd house-price-predictor

2. Install dependencies

pip install -r requirements.txt

3. Run the Streamlit app

streamlit run housePrice.py

⸻

📦 Requirements

The requirements.txt contains:

streamlit
pandas
numpy
scikit-learn

These are needed to:
	•	Build the UI (Streamlit)
	•	Handle data (Pandas, NumPy)
	•	Train the ML model (Scikit‑Learn)
