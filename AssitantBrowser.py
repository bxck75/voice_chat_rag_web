import tkinter as tk
from tkinter import messagebox, ttk
import requests
from typing import List, Union
import os
import tempfile
import warnings
import faiss
from typing import List, Dict, Any, Optional, Union
from git import Repo
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import requests
from rich import print as rp
from dotenv import load_dotenv, find_dotenv
import speech_recognition
from TTS.api import TTS
from playsound import playsound
from hugchat.login import Login
from hugchat import hugchat
from hugchat.hugchat import ChatBot,Conversation
from hugchat.types.assistant import Assistant
from hugchat.message import Message,MessageStatus,ModelOverloadedError
from hugchat.types.tool import Tool
from hugchat.types.file import File
from hugchat.types.model import Model
from hugchat.types.message import MessageNode, Conversation
from FaissStorage import LLMChatBot,AdvancedVectorStore
# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

# Import system prompts
from system_prompts import __all__ as prompts
rp(dir(Assistant))
rp(dir(File))
rp(dir(Tool))
rp(dir(MessageNode))
rp(dir(Conversation))
rp(dir(ChatBot))


class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HugChat Assistant Browser")
        self.hugchat = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))

        self.create_widgets()

    def create_widgets(self):
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True)

        # Create frames for each tab
        self.create_assistant_tab()

    def create_assistant_tab(self):
        self.assistant_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.assistant_tab, text="Assistants")

        self.search_frame = tk.Frame(self.assistant_tab)
        self.search_frame.pack(pady=10)

        self.search_label = tk.Label(self.search_frame, text="Search Assistant:")
        self.search_label.pack(side=tk.LEFT, padx=5)

        self.search_entry = tk.Entry(self.search_frame)
        self.search_entry.pack(side=tk.LEFT, padx=5)

        self.field_selector = ttk.Combobox(self.search_frame, values=["Name", "ID"])
        self.field_selector.set("Name")
        self.field_selector.pack(side=tk.LEFT, padx=5)

        self.search_button = tk.Button(self.search_frame, text="Search", command=self.search_assistant)
        self.search_button.pack(side=tk.LEFT, padx=5)

        self.page_frame = tk.Frame(self.assistant_tab)
        self.page_frame.pack(pady=10)

        self.page_label = tk.Label(self.page_frame, text="Browse Assistants by Page:")
        self.page_label.pack(side=tk.LEFT, padx=5)

        self.page_entry = tk.Entry(self.page_frame)
        self.page_entry.pack(side=tk.LEFT, padx=5)

        self.page_button = tk.Button(self.page_frame, text="Go", command=self.get_assistants_by_page)
        self.page_button.pack(side=tk.LEFT, padx=5)

        self.assistant_listbox = tk.Listbox(self.assistant_tab, width=100, height=15)
        self.assistant_listbox.pack(pady=10)
        self.assistant_listbox.bind('<<ListboxSelect>>', self.show_assistant_details)

        self.details_text = tk.Text(self.assistant_tab, width=100, height=15)
        self.details_text.pack(pady=10)



    def search_assistant(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a search query.")
            return

        field = self.field_selector.get()
        try:
            if field == "Name":
                assistant = self.hugchat.chatbot.search_assistant(assistant_name=query)
            elif field == "ID":
                assistant = self.hugchat.chatbot.search_assistant(assistant_id=query)
            else:
                assistant = None

            if assistant:
                self.assistant_listbox.delete(0, tk.END)
                self.assistant_listbox.insert(tk.END, f"{assistant.name} (by {assistant.author})")
                self.details_text.delete('1.0', tk.END)
                self.details_text.insert(tk.END, self.format_assistant_details(assistant))
            else:
                messagebox.showinfo("No Result", "No assistant found with that query.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_assistants_by_page(self):
        page = self.page_entry.get().strip()
        if not page.isdigit():
            messagebox.showwarning("Input Error", "Please enter a valid page number.")
            return

        try:
            assistants = self.hugchat.get_assistants(int(page))
            self.assistant_listbox.delete(0, tk.END)
            for assistant in assistants:
                self.assistant_listbox.insert(tk.END, f"{assistant.name} (by {assistant.author})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def get_conversations(self) -> List[Conversation]:
        return self.chatbot.get_conversation_list()

    def get_models(self) -> List[Model]:
        return self.chatbot.chatbot.get_available_llm_models()
    
    def get_active_model_id(self) -> List[Model]:
        return self.chatbot.get_active_llm_index()

    def get_assistant_by_id(self, id: str) -> Assistant:
        return self.chatbot.search_assistant(id)

    def send_message(self, conversation_id: str, message: str) -> Message:
        return self.chatbot.send_message(conversation_id, message)

    def get_current_assistent_id(self) -> Tool:
        return hugchat.Assistant.assistant_id # '65bd6d583140495b7e30f744'

    def show_assistant_details(self, event):
        selected_index = self.assistant_listbox.curselection()
        if selected_index:
            assistant = self.assistant_listbox.get(selected_index[0])
            self.details_text.delete('1.0', tk.END)
            self.details_text.insert(tk.END, assistant)

    
if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()
