# Tweet Filtering System 🚀
A machine learning-based **tweet classification system** that filters tweets based on **relevance and urgency** using **Logistic Regression, TF-IDF vectorization, and Sentiment Analysis**.  

## 🔹 Features
✅ Classifies tweets as **Informative or Not Informative**  
✅ Uses **TF-IDF vectorization** to extract important features from tweets  
✅ **Logistic Regression model** for classification with class balancing  
✅ Sentiment analysis using **TextBlob** (Urgent, Neutral, Not Urgent)  
✅ Stores trained models using **Pickle** for future predictions  
✅ Handles missing values and incorrect inputs gracefully  

##🛠 Tech Stack**  
- **Python** 🐍  
- **Pandas, NumPy** (Data Handling)  
- **Scikit-learn** (ML Model, TF-IDF, Label Encoding)  
- **TextBlob** (Sentiment Analysis)  
- **Pickle** (Model Storage)  

##📌 Installation & Setup
1️⃣ Clone this repository
```
git clone https://github.com/abyteofaditya/Tweet-Filtering-System.git
cd Tweet-Filtering-System
```
2️⃣ Install dependencies 
```
pip install pandas numpy scikit-learn textblob
```
3️⃣ Prepare dataset (CSV file required)  
- Ensure your dataset contains:  
  - `Actual_unr_tweets` (Tweet text)  
  - `category` (Informative/Not Informative)  
  - `confidence_score` (Relevance score)  
  - Any missing values will be handled automatically  

## 🚀 How to Run
### 1️⃣ Train the Model
```python
from TWEET_FILTERING_SYSTEM import training
training("your_dataset_filename")  # Without .csv extension
```
### 2️⃣ Predict & Filter Tweets
```python
from TWEET_FILTERING_SYSTEM import Predicting_output_for_user
Predicting_output_for_user("your_dataset_filename")  
```
### 3️⃣ Sentiment Analysis
The system will classify tweets as:  
- 🔴 Urgent(Negative sentiment)  
- 🟢 Not Urgent (Positive sentiment)  
- ⚪ Neutral 

## 📊 Sample Output  
```
Accuracy of the dataset is: 0.86  
The trained model appended successfully.  
Mapping of categories to encoded values:  
- Informative: 1  
- Not Informative: 0  
```

## 📝 Future Improvements 
🔹 Integrate deep learning (LSTMs, BERT) for better accuracy  
🔹 Deploy as a Flask API for real-time tweet filtering  
🔹 Improve sentiment analysis with VADER or fine-tuned models

## 📌 Contributing 
Feel free to fork this repo, suggest improvements, or report issues!  

## 📩 Contact 
💡 Created by Aditya Sarohaa  
📧 adityasarohaa55@gmail.com  
🔗 linkedin.com/in/aditya-sarohaa-345336323  

📜 License & Usage
📜 Copyright © 2024 Aditya Sarohaa
🔹 This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) License – see the LICENSE file for details.
🔹 You may view this code, but you may NOT use, modify, distribute, or profit from it without explicit permission from the author.
