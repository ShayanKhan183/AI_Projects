import tkinter as tk
from tkinter import messagebox, ttk
import pickle

# Load model and vectorizer
try:
    with open("count_vactor_email.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    with open("spam_email_detactor.pickle", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    messagebox.showerror("Error", "Model files not found. Run 'train_model.py' first.")
    exit()

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.animate_title()

    def setup_window(self):
        self.root.title("🎀 Advanced Spam Email Detector")
        self.root.geometry("700x600")
        self.root.resizable(False, False)
        self.root.configure(bg='#FFE5F1')
        self.root.eval('tk::PlaceWindow . center')

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=10)

    def create_widgets(self):
        self.title = tk.Label(self.root, text="🎀 Advanced Spam Email Detector 🎀",
                              font=('Helvetica', 22, 'bold'), bg='#FFE5F1', fg='#D63384')
        self.title.pack(pady=10)

        subtitle = tk.Label(self.root, text="✨ AI-powered spam protection ✨",
                            font=('Arial', 12, 'italic'), bg='#FFE5F1', fg='#B85450')
        subtitle.pack()

        input_label = tk.Label(self.root, text="📧 Enter Email Content:",
                               font=('Arial', 14, 'bold'), bg='#FFE5F1', fg='#8B4789')
        input_label.pack(pady=(20, 5))

        self.input_text = tk.Text(self.root, height=10, width=80, wrap='word',
                                  bg='#FFF0F5', fg='#333333', font=('Arial', 11),
                                  borderwidth=2, relief='groove')
        self.input_text.pack(padx=20)

        # Button frame
        button_frame = tk.Frame(self.root, bg='#FFE5F1')
        button_frame.pack(pady=15)

        self.check_btn = tk.Button(button_frame, text="🔍 Check for Spam",
                                   bg='#FF69B4', fg='white',
                                   font=('Arial Rounded MT Bold', 14, 'bold'),
                                   padx=20, pady=10,
                                   bd=0, relief='flat', cursor='hand2',
                                   activebackground='#FF1493',
                                   activeforeground='white',
                                   command=self.check_spam)
        self.check_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear",
                                   bg='#DDA0DD', fg='white',
                                   font=('Arial Rounded MT Bold', 13),
                                   padx=20, pady=10,
                                   bd=0, relief='flat', cursor='hand2',
                                   activebackground='#BA55D3',
                                   activeforeground='white',
                                   command=self.clear_text)
        self.clear_btn.grid(row=0, column=1, padx=10)

        # Hover effects
        self.check_btn.bind("<Enter>", lambda e: self.check_btn.config(bg='#FF1493'))
        self.check_btn.bind("<Leave>", lambda e: self.check_btn.config(bg='#FF69B4'))
        self.clear_btn.bind("<Enter>", lambda e: self.clear_btn.config(bg='#BA55D3'))
        self.clear_btn.bind("<Leave>", lambda e: self.clear_btn.config(bg='#DDA0DD'))

        self.output_label = tk.Label(self.root, text="🤖 Ready to analyze your email!",
                                     font=('Arial Rounded MT Bold', 16, 'bold'),
                                     bg='#FFE5F1', fg='#8B4789')
        self.output_label.pack(pady=20)

        self.status_bar = tk.Label(self.root, text="Status: Ready", bg='#FFE5F1', fg='#8B4789',
                                   anchor='w', font=('Arial', 10))
        self.status_bar.pack(fill='x', side='bottom', padx=10, pady=(10, 5))

    def animate_title(self):
        colors = ['#D63384', '#FF69B4', '#FF1493', '#DA70D6', '#DDA0DD']
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
        self.output_label.config(text="🤖 Ready to analyze your email!",
                                 fg='#8B4789', bg='#FFE5F1')
        self.status_bar.config(text="Status: Ready")

    def check_spam(self):
        email = self.input_text.get("1.0", tk.END).strip()
        if not email:
            messagebox.showwarning("Input Error", "Please enter some email content.")
            return
        self.output_label.config(text="🔄 Processing...", fg='#8B4789')
        self.status_bar.config(text="Status: Analyzing...")
        self.root.update_idletasks()
        self.root.after(100, lambda: self.process_email(email))

    def process_email(self, email):
        try:
            X = vectorizer.transform([email])
            result = model.predict(X)
            prediction = int(result[0])  # Ensure it's an integer

            print(f"Raw prediction: {prediction}")
            print(f"Model classes: {getattr(model, 'classes_', 'N/A')}")

            if prediction == 0:
                msg = "⚠️ SPAM DETECTED!"
                color = '#DC143C'
                bg = '#FFE4E1'
                self.status_bar.config(text="Status: Spam detected!")
            elif prediction == 1:
                msg = "✅ EMAIL IS SAFE!"
                color = '#228B22'
                bg = '#F0FFF0'
                self.status_bar.config(text="Status: Email is safe.")
            else:
                msg = f"❓ UNKNOWN RESULT: {prediction}"
                color = '#FF8C00'
                bg = '#FFF8DC'
                self.status_bar.config(text=f"Status: Unknown result.")

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
    app = SpamDetectorGUI(root)
    root.mainloop()
