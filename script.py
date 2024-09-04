import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import xgboost as xgb

def select_file():
    global dataset
    file_path = filedialog.askopenfilename()
    if file_path:
        file_display_label.config(text=f"Selected file: {file_path}")
        try:
            dataset = pd.read_csv(file_path)
            process_missing_data()
            visualize_button.config(state=tk.NORMAL)
            model_dropdown.config(state=tk.NORMAL)
            train_button.config(state=tk.NORMAL)
            predict_button.config(state=tk.NORMAL)
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")

def process_missing_data():
    global dataset
    if dataset is not None:
        total_na = dataset.isnull().sum().sum()
        if total_na > 0:
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            non_numeric_cols = dataset.select_dtypes(exclude=[np.number]).columns

            dataset[numeric_cols] = dataset[numeric_cols].apply(lambda col: col.fillna(col.mode()[0]))
            dataset[non_numeric_cols] = dataset[non_numeric_cols].apply(lambda col: col.fillna(col.mode()[0]))
        else:
            messagebox.showinfo("Info", "No missing data found.")

def show_data_visualization():
    global dataset
    if dataset is not None:
        sns.countplot(x='Label', data=dataset)
        plt.title('Distribution of Classification')
        plt.show()

def train_models():
    global dataset, xgb_clf, dt_clf, rf_clf
    if dataset is not None:
        x = dataset.drop(["Label"], axis=1)
        y = dataset['Label']
  
        y = y.map({'goodware': 0, 'malware': 1})
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

        results_output = {}
        selected_model = selected_model_var.get()

        if selected_model == 'xgb' or selected_model == 'all':
            xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(x_train, y_train)
            xgb_predictions = xgb_clf.predict(x_test)
            xgb_accuracy = accuracy_score(xgb_predictions, y_test)
            results_output['XGBoost'] = {'accuracy': xgb_accuracy, 'report': classification_report(y_test, xgb_predictions)}

        if selected_model == 'dt' or selected_model == 'all':
            max_depth_value = simpledialog.askstring("Input", "Enter max depth for Decision Tree:")
            if max_depth_value:
                max_depth_value = int(max_depth_value.strip())
                dt_clf = DecisionTreeClassifier(max_depth=max_depth_value)
                dt_clf.fit(x_train, y_train)
                dt_predictions = dt_clf.predict(x_test)
                dt_accuracy = accuracy_score(dt_predictions, y_test)
                results_output['Decision Tree'] = {'accuracy': dt_accuracy, 'report': classification_report(y_test, dt_predictions)}

        if selected_model == 'rf' or selected_model == 'all':
            num_estimators = simpledialog.askstring("Input", "Enter number of trees for Random Forest:")
            if num_estimators:
                num_estimators = int(num_estimators.strip())
                rf_clf = RandomForestClassifier(n_estimators=num_estimators, random_state=10)
                rf_clf.fit(x_train, y_train)
                rf_predictions = rf_clf.predict(x_test)
                rf_accuracy = accuracy_score(rf_predictions, y_test)
                results_output['Random Forest'] = {'accuracy': rf_accuracy, 'report': classification_report(y_test, rf_predictions)}

        show_results_window(results_output)

def show_results_window(results):
    results_window = tk.Toplevel()
    results_window.title("Model Training Results")

    results_text = tk.Text(results_window, height=20, width=80)
    results_text.pack(padx=10, pady=10)

    for model, metrics in results.items():
        results_text.insert(tk.END, f"{model}:\n")
        results_text.insert(tk.END, f"Accuracy: {metrics['accuracy']}\n")
        results_text.insert(tk.END, f"{metrics['report']}\n\n")

    results_text.config(state=tk.DISABLED)

def predict_new_input():
    global xgb_clf, dt_clf, rf_clf, dataset
    if dataset is not None:
        input_values = simpledialog.askstring("Input", "Enter the new input values (comma-separated):")
        if input_values:
            input_data = np.array([float(i) for i in input_values.split(',')]).reshape(1, -1)
            selected_model = selected_model_var.get()

            prediction_results = {}

            if selected_model == 'xgb' or selected_model == 'all':
                xgb_prediction = xgb_clf.predict(input_data)
                prediction_results['XGBoost'] = 'malware' if xgb_prediction[0] == 1 else 'goodware'

            if selected_model == 'dt' or selected_model == 'all':
                dt_prediction = dt_clf.predict(input_data)
                prediction_results['Decision Tree'] = 'malware' if dt_prediction[0] == 1 else 'goodware'

            if selected_model == 'rf' or selected_model == 'all':
                rf_prediction = rf_clf.predict(input_data)
                prediction_results['Random Forest'] = 'malware' if rf_prediction[0] == 1 else 'goodware'

            show_prediction_results(prediction_results)

def show_prediction_results(predictions):
    results_window = tk.Toplevel()
    results_window.title("Prediction Results")

    results_text = tk.Text(results_window, height=10, width=50)
    results_text.pack(padx=10, pady=10)

    for model, prediction in predictions.items():
        results_text.insert(tk.END, f"{model}: {prediction}\n")

    results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Malware Detection")

    dataset = None
    xgb_clf = None
    dt_clf = None
    rf_clf = None

    file_button = tk.Button(root, text="Select Dataset", command=select_file)
    file_button.pack(anchor='w', padx=10, pady=5)

    file_display_label = tk.Label(root, text="")
    file_display_label.pack(anchor='w', padx=10)

    visualize_button = tk.Button(root, text="Visualize Data", command=show_data_visualization, state=tk.DISABLED)
    visualize_button.pack(anchor='w', padx=10, pady=5)

    selected_model_var = tk.StringVar()
    selected_model_var.set("xgb")
    model_dropdown = ttk.Combobox(root, textvariable=selected_model_var, values=["xgb", "dt", "rf", "all"], state=tk.DISABLED)
    model_dropdown.pack(anchor='w', padx=10, pady=5)

    train_button = tk.Button(root, text="Train Models", command=train_models, state=tk.DISABLED)
    train_button.pack(anchor='w', padx=10, pady=5)

    predict_button = tk.Button(root, text="Predict New Input", command=predict_new_input, state=tk.DISABLED)
    predict_button.pack(anchor='w', padx=10, pady=5)

    root.mainloop()


#0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1
