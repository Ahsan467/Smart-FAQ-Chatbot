import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import nltk
import re
import sys
import tkinter as tk
from tkinter import scrolledtext

# --------- CONFIG ----------
TFIDF_THRESHOLD = 0.7
FUZZY_THRESHOLD = 70
SHOW_DEBUG = False
# --------------------------

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --------------------------
# Preprocess function
# --------------------------
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --------------------------
# Load FAQs
# --------------------------
try:
    faqs = pd.read_csv("faqs.csv", dtype=str, keep_default_na=False, quotechar='"')
except Exception as e:
    print("Error loading faqs.csv:", e)
    sys.exit(1)

if "question" not in faqs.columns or "answer" not in faqs.columns:
    print("faqs.csv must have 'question' and 'answer' columns.")
    sys.exit(1)

faqs = faqs.dropna(subset=["question", "answer"])
faqs["question"] = faqs["question"].astype(str)
faqs["answer"] = faqs["answer"].astype(str)
faqs["q_proc"] = faqs["question"].apply(preprocess)

if faqs["q_proc"].str.strip().eq("").all():
    print("Error: all processed questions are empty.")
    sys.exit(1)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faqs["q_proc"])

# --------------------------
# Custom generic fallback intents
# --------------------------
generic_intents = {
    "what do you do": "🤖 I am an FAQ Chatbot. I can answer your questions about services and policies.",
    "who are you": "🤖 I am an FAQ Chatbot created to assist you.",
    "introduce yourself": "👋 Hello! I'm an FAQ Chatbot that answers your queries.",
    "your father": "❌ I am an AI, I don't have a father."
}

# --------------------------
# Matching function
# --------------------------
def get_answer(user_input: str) -> str:
    # --- check generic intents first ---
    for intent, response in generic_intents.items():
        if intent in user_input.lower():
            return response

    questions = re.split(r'[?.!]', user_input)
    responses = []

    for q in questions:
        q = q.strip()
        if not q:
            continue

        user_proc = preprocess(q)
        if not user_proc:
            responses.append("❌ Sorry, I couldn't understand the question.")
            continue

        # --- TF-IDF ---
        user_vec = vectorizer.transform([user_proc])
        similarities = cosine_similarity(user_vec, X)
        idx = similarities.argmax()
        score = similarities[0, idx]

        if score >= TFIDF_THRESHOLD and faqs.loc[idx, "q_proc"].strip() != "":
            responses.append(faqs.loc[idx, "answer"])
            continue

        # --- Fuzzy ---
        q_choices = faqs["q_proc"].tolist()
        fuzzy_best = process.extractOne(user_proc, q_choices, scorer=fuzz.token_sort_ratio)

        if fuzzy_best and fuzzy_best[1] >= FUZZY_THRESHOLD:
            try:
                matched_idx = q_choices.index(fuzzy_best[0])
                responses.append(faqs.loc[matched_idx, "answer"])
                continue
            except ValueError:
                pass

        responses.append("🤷 Sorry, I don't know the answer.")

    return "\n".join(responses)

# --------------------------
# Tkinter Chat UI
# --------------------------
def send_message():
    user_input = entry.get().strip()
    if not user_input:
        return
    chat_area.insert(tk.END, f"\n🧑 You: {user_input}\n", "user")
    entry.delete(0, tk.END)

    if user_input.lower() == "exit":
        chat_area.insert(tk.END, "\n🤖 Chatbot: Goodbye! 👋\n", "bot")
        root.after(1500, root.destroy)
        return

    bot_response = get_answer(user_input)
    chat_area.insert(tk.END, f"\n🤖 Chatbot: {bot_response}\n", "bot")
    chat_area.see(tk.END)

# --------------------------
# Attractive UI Setup
# --------------------------
root = tk.Tk()
root.title("💡 Smart FAQ Chatbot")
root.geometry("700x570")
root.config(bg="#2C2F48")  # deep stylish bg

# Header
header = tk.Label(root, text="✨ Smart FAQ Chatbot ✨", 
                  font=("Comic Sans MS", 20, "bold"), 
                  bg="#1100FF", fg="white", pady=12)
header.pack(fill=tk.X)

# Chat area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, 
                                      font=("Verdana", 12),
                                      bg="#f4f4f9", fg="#2C3E50")
chat_area.pack(padx=12, pady=12, fill=tk.BOTH, expand=True)
chat_area.insert(tk.END, "🤖 Welcome! Ask me anything or type 'exit' to quit.\n", "bot")

# Style tags (with background colors)
chat_area.tag_config("user", foreground="#ffffff", background="#3B82F6", font=("Verdana", 12, "bold"))  # Blue bg + white text
chat_area.tag_config("bot", foreground="#155724", background="#D4EDDA", font=("Verdana", 12, "bold"))   # Green text + light green bg

# Bottom frame
frame = tk.Frame(root, bg="#2C2F48")
frame.pack(padx=12, pady=8, fill=tk.X)

entry = tk.Entry(frame, font=("Verdana", 12), bg="#ffffff", fg="#2C3E50", relief="solid", bd=2)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=8)

send_btn = tk.Button(frame, text="🚀 Send", command=send_message,
                     font=("Verdana", 12, "bold"),
                     bg="#1605FF", fg="white",
                     activebackground="#001AFF",
                     relief="flat", padx=20, pady=6,
                     cursor="hand2")
send_btn.pack(side=tk.RIGHT)

root.mainloop()
