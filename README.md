# 💼 Employee Salary Prediction App

This Streamlit web application predicts whether an employee earns more than **$50K/year** based on demographic and work-related inputs.

---

## 🚀 Features

- Clean and modern web interface
- Predicts income level using a **Random Forest Classifier**
- Input fields include age, education, occupation, and more
- Dynamic dropdowns that update from your data
- Easy to deploy on **Streamlit Cloud**

---

## 📁 Files

| File/Folder     | Description                                      |
|------------------|--------------------------------------------------|
| `app.py`         | Main Streamlit app file                         |
| `adult 3.csv`    | Dataset used for training                       |
| `banner.jpg`     | Optional banner image for homepage              |
| `README.md`      | This file                                       |
| `requirements.txt` | Required Python packages for Streamlit Cloud |

---

## 🛠️ How to Run Locally

1. **Install dependencies** (create a virtual environment if you prefer):

    ```bash
    pip install -r requirements.txt
    ```

2. **Place your files**:
    - `app.py`
    - `adult 3.csv`
    - (Optional) `banner.jpg` for homepage image

3. **Run the app**:

    ```bash
    streamlit run app.py
    ```

4. Visit `http://localhost:8501` in your browser.

---

## 🌐 Deployment on Streamlit Cloud

1. Push all files to your **GitHub repo** (including `app.py`, `adult 3.csv`, `banner.jpg`, `requirements.txt`)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select `app.py` as the main file and click **Deploy**

---

## 📸 Preview

- **Homepage**: Shows a welcome image with a button "🔍 Predict Salary"
- **Prediction Page**: Takes input fields and shows the predicted income class.

---

## 🤝 Credits

Built with ❤️ using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).

---

## 📄 License

This project is open-source and free to use for educational and personal purposes.

