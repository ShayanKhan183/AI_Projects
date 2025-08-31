import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import os

# Load model and vectorizer with base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    messagebox.showerror("Error", "❌ Model files not found.\nRun 'train_model.py' first.")
    exit(1)

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.animate_title()

    def setup_window(self):
        self.root.title("🌟 Sentiment Analyzer")
        self.root.geometry("720x640")
        self.root.resizable(False, False)
        self.root.configure(bg='#F0F8FF')
        self.root.eval('tk::PlaceWindow . center')

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=10)

    def create_widgets(self):
        self.title = tk.Label(self.root, text="🌟 Sentiment Analyzer 🌟",
                              font=('Helvetica', 22, 'bold'), bg='#F0F8FF', fg='#4B0082')
        self.title.pack(pady=10)

        subtitle = tk.Label(self.root, text="🔍 Analyze if your text is Positive or Negative",
                            font=('Arial', 12, 'italic'), bg='#F0F8FF', fg='#8B008B')
        subtitle.pack()

        input_label = tk.Label(self.root, text="✏️ Enter Your Text Below:",
                               font=('Arial', 14, 'bold'), bg='#F0F8FF', fg='#4B0082')
        input_label.pack(pady=(20, 5))

        self.input_text = tk.Text(self.root, height=10, width=85, wrap='word',
                                  bg='#F5F5FF', fg='#333333', font=('Arial', 11),
                                  borderwidth=2, relief='groove')
        self.input_text.pack(padx=20)

        self.char_count = tk.Label(self.root, text="Characters: 0", bg='#F0F8FF', fg='#888888',
                                   font=('Arial', 9, 'italic'))
        self.char_count.pack(pady=5)
        self.input_text.bind("<KeyRelease>", self.update_char_count)

        button_frame = tk.Frame(self.root, bg='#F0F8FF')
        button_frame.pack(pady=15)

        self.analyze_btn = tk.Button(button_frame, text="📈 Analyze Sentiment",
                                     bg='#4B0082', fg='white',
                                     font=('Arial Rounded MT Bold', 14, 'bold'),
                                     padx=20, pady=10,
                                     bd=0, relief='flat', cursor='hand2',
                                     activebackground='#6A0DAD',
                                     activeforeground='white',
                                     command=self.analyze_sentiment)
        self.analyze_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = tk.Button(button_frame, text="🧹 Clear",
                                   bg='#9370DB', fg='white',
                                   font=('Arial Rounded MT Bold', 13),
                                   padx=20, pady=10,
                                   bd=0, relief='flat', cursor='hand2',
                                   activebackground='#8A2BE2',
                                   activeforeground='white',
                                   command=self.clear_text)
        self.clear_btn.grid(row=0, column=1, padx=10)

        # Hover effects
        self.analyze_btn.bind("<Enter>", lambda e: self.analyze_btn.config(bg='#6A0DAD'))
        self.analyze_btn.bind("<Leave>", lambda e: self.analyze_btn.config(bg='#4B0082'))
        self.clear_btn.bind("<Enter>", lambda e: self.clear_btn.config(bg='#8A2BE2'))
        self.clear_btn.bind("<Leave>", lambda e: self.clear_btn.config(bg='#9370DB'))

        self.output_label = tk.Label(self.root, text="🤖 Awaiting input...",
                                     font=('Arial Rounded MT Bold', 16, 'bold'),
                                     bg='#F0F8FF', fg='#4B0082')
        self.output_label.pack(pady=20)

        self.status_bar = tk.Label(self.root, text="Status: Ready", bg='#F0F8FF', fg='#4B0082',
                                   anchor='w', font=('Arial', 10))
        self.status_bar.pack(fill='x', side='bottom', padx=10, pady=(10, 5))

    def update_char_count(self, event=None):
        text = self.input_text.get("1.0", tk.END)
        self.char_count.config(text=f"Characters: {len(text.strip())}")

    def animate_title(self):
        colors = ['#4B0082', '#6A0DAD', '#8A2BE2']
        current_color = colors[0]
        def change_color():
            nonlocal current_color
            idx = colors.index(current_color)
            next_color = colors[(idx + 1) % len(colors)]
            self.title.config(fg=next_color)
            current_color = next_color
            self.root.after(1500, change_color)
        change_color()

    def animate_output(self):
        def pulse(count=0):
            if count < 6:
                new_size = 16 + (count % 2)
                self.output_label.config(font=('Arial Rounded MT Bold', new_size, 'bold'))
                self.root.after(200, lambda: pulse(count + 1))
        pulse()

    def clear_text(self):
        self.input_text.delete("1.0", tk.END)
        self.output_label.config(text="🤖 Awaiting input...", fg='#4B0082', bg='#F0F8FF')
        self.status_bar.config(text="Status: Ready")
        self.char_count.config(text="Characters: 0")

    def analyze_sentiment(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return
        self.output_label.config(text="🔄 Analyzing...", fg='#4B0082')
        self.status_bar.config(text="Status: Analyzing...")
        self.root.update_idletasks()
        self.root.after(100, lambda: self.process_input(text))

    def process_input(self, text):
        try:
            X = vectorizer.transform([text])
            result = model.predict(X)[0].lower()

            if result == "positive":
                msg = "😊 POSITIVE SENTIMENT!"
                color = '#228B22'
                bg = '#F0FFF0'
                self.status_bar.config(text="Status: Positive sentiment detected.")
            elif result == "negative":
                msg = "😠 NEGATIVE SENTIMENT!"
                color = '#B22222'
                bg = '#FFF0F0'
                self.status_bar.config(text="Status: Negative sentiment detected.")
            else:
                msg = f"🤔 UNKNOWN RESULT: {result}"
                color = '#FF8C00'
                bg = '#FFFFE0'
                self.status_bar.config(text="Status: Unknown sentiment.")

            self.output_label.config(text=msg, fg=color, bg=bg)
            self.output_label.update_idletasks()
            self.animate_output()
            messagebox.showinfo("Prediction Result", msg)

        except Exception as e:
            print("Error:", str(e))
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            self.output_label.config(text="❌ ERROR OCCURRED!", fg='#DC143C')
            self.status_bar.config(text="Status: Error occurred")

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
