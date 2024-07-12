import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import json,subprocess
import random
from TTS.api import TTS
class VoiceProfile:
    def __init__(self, name, voice):
        self.name = name
        self.voice = voice

    def to_dict(self):
        return {
            'name': self.name,
            'voice': self.voice
        }

    @classmethod
    def from_dict(cls, profile_dict):
        return cls(
            profile_dict['name'],
            profile_dict['voice']
        )

    def configure_tts(self):
        # Set Festival voice
        subprocess.run(["festival", "--tts", "(voice_" + self.voice + ")"])

class VoiceProfileManager:
    def __init__(self, filename="voice_profiles.json"):
        self.filename = filename
        self.profiles = []
        self.load_profiles()

    def load_profiles(self):
        try:
            with open(self.filename, 'r') as file:
                profiles_data = json.load(file)
                self.profiles = [VoiceProfile.from_dict(profile) for profile in profiles_data]
        except FileNotFoundError:
            print(f"File '{self.filename}' not found. Starting with an empty profile list.")
            self.profiles = []

    def save_profiles(self):
        profiles_data = [profile.to_dict() for profile in self.profiles]
        with open(self.filename, 'w') as file:
            json.dump(profiles_data, file, indent=4)
        print(f"Profiles saved to '{self.filename}'.")

    def add_profile(self, profile):
        self.profiles.append(profile)

    def generate_random_profile(self):
        name = f"Profile-{len(self.profiles) + 1}"
        voices = ["cmu_us_slt", "cmu_us_awb", "cmu_us_rms", "cmu_us_bdl"]  # Example Festival voices
        voice = random.choice(voices)
        new_profile = VoiceProfile(name, voice)
        self.add_profile(new_profile)
        return new_profile

    def list_profiles(self):
        if not self.profiles:
            return "No profiles found."
        else:
            profiles_list = []
            for idx, profile in enumerate(self.profiles, start=1):
                profiles_list.append(f"Profile {idx}: {profile.name} - Voice: {profile.voice}")
            return profiles_list

    def get_profile_by_name(self, profile_name):
        for profile in self.profiles:
            if profile.name == profile_name:
                return profile
        return None


class VoiceProfileTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Profile Manager")

        self.profile_manager = VoiceProfileManager()

        self.create_widgets()

    def create_widgets(self):
        # Frame for profile list and operations
        profile_frame = ttk.LabelFrame(self.root, text="Voice Profiles")
        profile_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

        # Listbox to display profiles
        self.profiles_listbox = tk.Listbox(profile_frame, width=50, height=10)
        self.profiles_listbox.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

        # Scrollbar for the listbox
        scrollbar = ttk.Scrollbar(profile_frame, orient=tk.VERTICAL, command=self.profiles_listbox.yview)
        scrollbar.grid(row=0, column=1, pady=10, sticky=tk.N+tk.S)
        self.profiles_listbox.config(yscrollcommand=scrollbar.set)

        # Button to generate random profile
        generate_btn = ttk.Button(profile_frame, text="Generate Random Profile", command=self.generate_random_profile)
        generate_btn.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W+tk.E)

        # Button to refresh profile list
        refresh_btn = ttk.Button(profile_frame, text="Refresh List", command=self.refresh_profiles_list)
        refresh_btn.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W+tk.E)

        # Frame for TTS operations
        tts_frame = ttk.LabelFrame(self.root, text="Text-to-Speech (TTS)")
        tts_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

        # Text entry for TTS input
        ttk.Label(tts_frame, text="Enter text to speak:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.tts_text_entry = ttk.Entry(tts_frame, width=50)
        self.tts_text_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W+tk.E)

        # Combobox to select profile for TTS
        ttk.Label(tts_frame, text="Select profile:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.profile_combobox = ttk.Combobox(tts_frame, width=48, state="readonly")
        self.profile_combobox.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W+tk.E)

        # Button to speak text
        speak_btn = ttk.Button(tts_frame, text="Speak", command=self.speak_text)
        speak_btn.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W+tk.E)

        # Populate initial profiles list
        self.refresh_profiles_list()

    def refresh_profiles_list(self):
        # Clear current listbox and combobox
        self.profiles_listbox.delete(0, tk.END)
        self.profile_combobox['values'] = []

        # Load profiles from manager
        profiles = self.profile_manager.list_profiles()
        if profiles:
            for profile in profiles:
                self.profiles_listbox.insert(tk.END, profile)
                self.profile_combobox['values'] += (profile.split(':')[0],)  # Add profile name to combobox options

    def generate_random_profile(self):
        new_profile = self.profile_manager.generate_random_profile()
        messagebox.showinfo("Profile Generated", f"Generated new profile: {new_profile.name}")
        self.refresh_profiles_list()

    def speak_text(self):
        text_to_speak = self.tts_text_entry.get().strip()
        selected_profile_name = self.profile_combobox.get().strip()

        if not text_to_speak:
            messagebox.showwarning("Input Required", "Please enter text to speak.")
            return

        if not selected_profile_name:
            messagebox.showwarning("Profile Required", "Please select a profile.")
            return

        profile = self.profile_manager.get_profile_by_name(selected_profile_name)
        if profile:
            profile.configure_tts()
            subprocess.Popen(["festival", "--tts"], stdin=subprocess.PIPE).communicate(bytes(text_to_speak, 'utf-8'))
            messagebox.showinfo("Text-to-Speech", f"Text: {text_to_speak}\nProfile: {profile.name}\nVoice: {profile.voice}")
        else:
            messagebox.showerror("Profile Not Found", f"Profile '{selected_profile_name}' not found.")

# Main program
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceProfileTool(root)
    root.mainloop()
""" Explanation:
Integration with piper_tts: This example uses the TTS class from piper_tts for text-to-speech synthesis. The configure_tts method in VoiceProfile class is used to set parameters (language, pitch, speaking_rate) on the TTS engine before synthesizing speech.

Tkinter GUI: The GUI interface (VoiceProfileTool class) is built using Tkinter widgets (Listbox, Entry, Combobox, Button, etc.) to manage voice profiles (list, generate random profile) and perform TTS (enter text, select profile, speak).

Profile Management: VoiceProfileManager handles loading/saving profiles from/to JSON file, generating random profiles, listing profiles, and retrieving profiles by name.

Handling TTS Output: After synthesizing speech with piper_tts, the example shows a message box with details about the synthesized text and the selected profile.
 """
