import tkinter as tk    #gui 
from tkinter import ttk, messagebox, filedialog
import pandas as pd     # data handling
import seaborn as sb      #data visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg    #data visualization
import sklearn.utils as u     #for machine learning models)
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn       #for machine learning models)
import numpy as np    #data handling
import warnings as w
w.filterwarnings('ignore')

class StudentPerformanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.data = None
        self.models = {}
        self.accuracies = {}
        self.is_trained = False
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Student Performance Prediction System", 
                              font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Data Loading and Visualization
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data & Visualization")
        
        # Tab 2: Model Training and Results
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model Training")
        
        # Tab 3: Prediction
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_tab, text="Make Prediction")
        
        self.create_data_tab()
        self.create_model_tab()
        self.create_predict_tab()
        
    def create_data_tab(self):
        # Data loading section
        data_frame = tk.LabelFrame(self.data_tab, text="Data Loading", font=("Arial", 12, "bold"))
        data_frame.pack(fill='x', padx=10, pady=5)
        
        load_btn = tk.Button(data_frame, text="Load CSV File", command=self.load_data,
                            bg='#3498db', fg='white', font=("Arial", 10, "bold"))
        load_btn.pack(side='left', padx=10, pady=10)
        
        self.data_status = tk.Label(data_frame, text="No data loaded", 
                                   font=("Arial", 10), fg='#e74c3c')
        self.data_status.pack(side='left', padx=10)
        
        # Visualization section
        viz_frame = tk.LabelFrame(self.data_tab, text="Data Visualization", font=("Arial", 12, "bold"))
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Graph selection
        graph_control_frame = tk.Frame(viz_frame)
        graph_control_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(graph_control_frame, text="Select Graph Type:", font=("Arial", 10)).pack(side='left')
        
        self.graph_var = tk.StringVar(value="Class Count")
        graph_options = [
            "Class Count", "Class by Semester", "Class by Gender", 
            "Class by Nationality", "Class by Grade", "Class by Section",
            "Class by Topic", "Class by Stage", "Class by Absent Days"
        ]
        
        graph_combo = ttk.Combobox(graph_control_frame, textvariable=self.graph_var, 
                                  values=graph_options, state='readonly', width=20)
        graph_combo.pack(side='left', padx=10)
        
        show_graph_btn = tk.Button(graph_control_frame, text="Show Graph", 
                                  command=self.show_graph, bg='#2ecc71', fg='white',
                                  font=("Arial", 10, "bold"))
        show_graph_btn.pack(side='left', padx=10)
        
        # Graph display area
        self.graph_frame = tk.Frame(viz_frame)
        self.graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_model_tab(self):
        # Model training section
        train_frame = tk.LabelFrame(self.model_tab, text="Model Training", font=("Arial", 12, "bold"))
        train_frame.pack(fill='x', padx=10, pady=5)
        
        train_btn = tk.Button(train_frame, text="Train All Models", command=self.train_models,
                             bg='#e67e22', fg='white', font=("Arial", 12, "bold"))
        train_btn.pack(side='left', padx=10, pady=10)
        
        self.train_status = tk.Label(train_frame, text="Models not trained", 
                                    font=("Arial", 10), fg='#e74c3c')
        self.train_status.pack(side='left', padx=10)
        
        # Results section
        results_frame = tk.LabelFrame(self.model_tab, text="Model Results", font=("Arial", 12, "bold"))
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create text widget for results
        self.results_text = tk.Text(results_frame, height=25, width=80, 
                                   font=("Consolas", 10))
        scrollbar = tk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
        
    def create_predict_tab(self):
        # Input section
        input_frame = tk.LabelFrame(self.predict_tab, text="Input Features", font=("Arial", 12, "bold"))
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Create input fields
        fields_frame = tk.Frame(input_frame)
        fields_frame.pack(padx=10, pady=10)
        
        # First row
        row1 = tk.Frame(fields_frame)
        row1.pack(fill='x', pady=5)
        
        tk.Label(row1, text="Raised Hands:", font=("Arial", 10)).pack(side='left')
        self.raised_hands_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.raised_hands_var, width=10).pack(side='left', padx=5)
        
        tk.Label(row1, text="Visited Resources:", font=("Arial", 10)).pack(side='left', padx=(20,0))
        self.visited_resources_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.visited_resources_var, width=10).pack(side='left', padx=5)
        
        # Second row
        row2 = tk.Frame(fields_frame)
        row2.pack(fill='x', pady=5)
        
        tk.Label(row2, text="Discussions:", font=("Arial", 10)).pack(side='left')
        self.discussions_var = tk.StringVar()
        tk.Entry(row2, textvariable=self.discussions_var, width=10).pack(side='left', padx=5)
        
        tk.Label(row2, text="Absence Days:", font=("Arial", 10)).pack(side='left', padx=(20,0))
        self.absence_var = tk.StringVar(value="Under-7")
        absence_combo = ttk.Combobox(row2, textvariable=self.absence_var, 
                                   values=["Under-7", "Above-7"], state='readonly', width=10)
        absence_combo.pack(side='left', padx=5)
        
        # Predict button
        predict_btn = tk.Button(input_frame, text="Predict Performance", 
                               command=self.make_prediction, bg='#9b59b6', fg='white',
                               font=("Arial", 12, "bold"))
        predict_btn.pack(pady=10)
        
        # Results section
        pred_results_frame = tk.LabelFrame(self.predict_tab, text="Prediction Results", 
                                         font=("Arial", 12, "bold"))
        pred_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.pred_results_text = tk.Text(pred_results_frame, height=15, width=80, 
                                        font=("Arial", 11))
        pred_scrollbar = tk.Scrollbar(pred_results_frame, orient="vertical", 
                                     command=self.pred_results_text.yview)
        self.pred_results_text.configure(yscrollcommand=pred_scrollbar.set)
        
        self.pred_results_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        pred_scrollbar.pack(side='right', fill='y')
        
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.data_status.config(text=f"Data loaded successfully! Shape: {self.data.shape}", 
                                       fg='#27ae60')
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
                
    def show_graph(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        graph_type = self.graph_var.get()
        
        try:
            if graph_type == "Class Count":
                sb.countplot(x='Class', data=self.data, order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Count Graph')
            elif graph_type == "Class by Semester":
                sb.countplot(x='Semester', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Semester-wise Graph')
            elif graph_type == "Class by Gender":
                sb.countplot(x='gender', hue='Class', data=self.data, 
                           order=['M', 'F'], hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Gender-wise Graph')
            elif graph_type == "Class by Nationality":
                sb.countplot(x='NationalITy', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Nationality-wise Graph')
            elif graph_type == "Class by Grade":
                sb.countplot(x='GradeID', hue='Class', data=self.data, 
                           order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 
                                  'G-09', 'G-10', 'G-11', 'G-12'], 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Grade-wise Graph')
            elif graph_type == "Class by Section":
                sb.countplot(x='SectionID', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Section-wise Graph')
            elif graph_type == "Class by Topic":
                sb.countplot(x='Topic', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Topic-wise Graph')
            elif graph_type == "Class by Stage":
                sb.countplot(x='StageID', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Stage-wise Graph')
            elif graph_type == "Class by Absent Days":
                sb.countplot(x='StudentAbsenceDays', hue='Class', data=self.data, 
                           hue_order=['L', 'M', 'H'], ax=ax)
                ax.set_title('Marks Class Absent Days-wise Graph')
                
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create graph: {str(e)}")
            
    def train_models(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            # Prepare data (similar to original code)
            data_copy = self.data.copy()
            
            # Drop unnecessary columns
            columns_to_drop = ["gender", "StageID", "GradeID", "NationalITy", 
                              "PlaceofBirth", "SectionID", "Topic", "Semester", 
                              "Relation", "ParentschoolSatisfaction", 
                              "ParentAnsweringSurvey", "AnnouncementsView"]
            
            for col in columns_to_drop:
                if col in data_copy.columns:
                    data_copy = data_copy.drop(col, axis=1)
            
            # Shuffle data
            u.shuffle(data_copy)
            
            # Encode categorical variables
            for column in data_copy.columns:
                if data_copy[column].dtype == type(object):
                    le = pp.LabelEncoder()
                    data_copy[column] = le.fit_transform(data_copy[column])
            
            # Split data
            ind = int(len(data_copy) * 0.70)
            feats = data_copy.values[:, 0:4]
            lbls = data_copy.values[:, 4]
            
            feats_train = feats[0:ind]
            feats_test = feats[(ind+1):len(feats)]
            lbls_train = lbls[0:ind]
            lbls_test = lbls[(ind+1):len(lbls)]
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training Models...\n\n")
            self.root.update()
            
            # Train models
            models_info = [
                ("Decision Tree", tr.DecisionTreeClassifier()),
                ("Random Forest", es.RandomForestClassifier()),
                ("Perceptron", lm.Perceptron()),
                ("Logistic Regression", lm.LogisticRegression()),
                ("Neural Network", nn.MLPClassifier(activation="logistic"))
            ]
            
            for name, model in models_info:
                self.results_text.insert(tk.END, f"Training {name}...\n")
                self.root.update()
                
                model.fit(feats_train, lbls_train)
                predictions = model.predict(feats_test)
                
                # Calculate accuracy
                accuracy = sum(a == b for a, b in zip(lbls_test, predictions)) / len(lbls_test)
                
                # Store model and accuracy
                self.models[name] = model
                self.accuracies[name] = accuracy
                
                # Display results
                self.results_text.insert(tk.END, f"\n{name} Results:\n")
                self.results_text.insert(tk.END, f"Accuracy: {round(accuracy, 3)}\n")
                self.results_text.insert(tk.END, f"Classification Report:\n")
                self.results_text.insert(tk.END, m.classification_report(lbls_test, predictions))
                self.results_text.insert(tk.END, "\n" + "="*50 + "\n\n")
                self.root.update()
            
            self.is_trained = True
            self.train_status.config(text="All models trained successfully!", fg='#27ae60')
            messagebox.showinfo("Success", "All models trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train models: {str(e)}")
            
    def make_prediction(self):
        if not self.is_trained:
            messagebox.showwarning("Warning", "Please train models first!")
            return
            
        try:
            # Get input values
            raised_hands = int(self.raised_hands_var.get())
            visited_resources = int(self.visited_resources_var.get())
            discussions = int(self.discussions_var.get())
            absence = 1 if self.absence_var.get() == "Under-7" else 0
            
            # Create input array
            input_array = np.array([raised_hands, visited_resources, discussions, absence])
            
            # Clear results
            self.pred_results_text.delete(1.0, tk.END)
            self.pred_results_text.insert(tk.END, "Prediction Results:\n\n")
            
            # Make predictions with all models
            class_mapping = {0: "High (H)", 1: "Medium (M)", 2: "Low (L)"}
            
            for name, model in self.models.items():
                prediction = model.predict(input_array.reshape(1, -1))[0]
                predicted_class = class_mapping.get(prediction, "Unknown")
                accuracy = self.accuracies[name]
                
                self.pred_results_text.insert(tk.END, 
                    f"{name}:\n"
                    f"  Predicted Performance: {predicted_class}\n"
                    f"  Model Accuracy: {round(accuracy, 3)}\n\n")
            
            # Summary
            self.pred_results_text.insert(tk.END, "="*40 + "\n")
            self.pred_results_text.insert(tk.END, "Input Summary:\n")
            self.pred_results_text.insert(tk.END, f"Raised Hands: {raised_hands}\n")
            self.pred_results_text.insert(tk.END, f"Visited Resources: {visited_resources}\n")
            self.pred_results_text.insert(tk.END, f"Discussions: {discussions}\n")
            self.pred_results_text.insert(tk.END, f"Absence Days: {self.absence_var.get()}\n")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values!")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

def main():
    root = tk.Tk()
    app = StudentPerformanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()