import tkinter as tk
from tkinter import filedialog, scrolledtext
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

def generate_wordcloud():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        tk.messagebox.showerror("Error", "Please enter some text or select a file.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)

    if hasattr(generate_wordcloud, 'canvas'):
        generate_wordcloud.canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    generate_wordcloud.canvas = canvas

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text_input.delete("1.0", tk.END)
            text_input.insert(tk.END, text)

# Create main window
root = tk.Tk()
root.title("Word Cloud Generator")
root.geometry("800x600")

# Create and pack widgets
tk.Label(root, text="Enter text or select a file:").pack(pady=5)

text_input = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD)
text_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

tk.Button(button_frame, text="Select File", command=select_file).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Generate Word Cloud", command=generate_wordcloud).pack(side=tk.LEFT, padx=5)

root.mainloop()