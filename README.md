
# **🚀 Tweet Filtering System**  
🔍 *An AI-powered system that classifies tweets based on **relevance and urgency** using **Machine Learning, TF-IDF vectorization, and Sentiment Analysis**.*  

---

## **📌 Features**  
✅ **Classifies tweets as Informative or Not Informative**  
✅ **Uses TF-IDF vectorization** to extract meaningful words  
✅ **Logistic Regression Model** for classification  
✅ **Sentiment Analysis** (Urgent 🔴, Neutral ⚪, Not Urgent 🟢)  
✅ **Handles missing values & incorrect inputs gracefully**  
✅ **Trained models are saved for future predictions**  

---

## **🛠 Tech Stack**  
🔹 **Python** 🐍  
🔹 **Pandas, NumPy** (Data Handling)  
🔹 **Scikit-learn** (ML Model, TF-IDF, Label Encoding)  
🔹 **TextBlob** (Sentiment Analysis)  
🔹 **Pickle** (Model Storage)  

---

## **📂 Installation & Setup**  

### **1️⃣ Install Dependencies**  
```bash
pip install pandas numpy scikit-learn textblob
```

### **2️⃣ Prepare Your Dataset (CSV File Required)**  
Ensure your dataset contains the following columns:  
- **`Actual_unr_tweets`** → Tweet text  
- **`category`** → Informative / Not Informative  
- **`confidence_score`** → Relevance score  
📌 *Missing values will be handled automatically!*  

---

## **🚀 How to Run**  

### **1️⃣ Train the Model**  
```python
from TWEET_FILTERING_SYSTEM import training
training("your_dataset_filename")  # Without .csv extension
```

### **2️⃣ Predict & Filter Tweets**  
```python
from TWEET_FILTERING_SYSTEM import Predicting_output_for_user
Predicting_output_for_user("your_dataset_filename")  
```

### **3️⃣ Sentiment Analysis**  
The system will classify tweets as:  
- 🔴 **Urgent** (*Negative Sentiment*)  
- 🟢 **Not Urgent** (*Positive Sentiment*)  
- ⚪ **Neutral**  

---

## **📊 Sample Output**  
```
Accuracy of the dataset is: 0.86  
The trained model appended successfully.  
Mapping of categories to encoded values:  
- Informative: 1  
- Not Informative: 0  
```

---

## **📝 Future Improvements 🚀**  
🔹 **Integrate Deep Learning (LSTMs, BERT) for better accuracy**  
🔹 **Deploy as a Flask API for real-time tweet filtering**  
🔹 **Improve Sentiment Analysis with VADER or fine-tuned models**  

---

## **💡 Contributing**  
Feel free to **suggest improvements or report issues!**  

---

## **📩 Contact**  
💡 **Created by:** Aditya Sarohaa  
📧 **Email:** [adityasarohaa55@gmail.com](mailto:adityasarohaa55@gmail.com)  
🔗 **LinkedIn:** [linkedin.com/in/aditya-sarohaa-345336323](https://linkedin.com/in/aditya-sarohaa-345336323)  

---

## **📜 License & Usage**  
📜 **Copyright © 2024 Aditya Sarohaa**  
🔹 **Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) License** – see the [LICENSE](LICENSE) file for details.  
🔹 **You may view this code, but you may NOT use, modify, distribute, or profit from it without explicit permission from the author.**  

