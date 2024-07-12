import json
import random

class VoiceProfileManager:
    def __init__(self, filename="voice_profiles.json"):
        self.filename = filename
        self.profiles = []

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
        languages = ["en-US", "en-GB", "fr-FR", "es-ES"]  # Example languages
        language = random.choice(languages)
        pitch = round(random.uniform(0.8, 1.2), 2)
        speaking_rate = round(random.uniform(0.7, 1.3), 2)
        new_profile = VoiceProfile(name, language, pitch, speaking_rate)
        self.add_profile(new_profile)
        return new_profile

    def list_profiles(self):
        if not self.profiles:
            print("No profiles found.")
        else:
            for idx, profile in enumerate(self.profiles, start=1):
                print(f"Profile {idx}: {profile.name} - Language: {profile.language}, Pitch: {profile.pitch}, Speaking Rate: {profile.speaking_rate}")

# Example usage:
if __name__ == "__main__":
    manager = VoiceProfileManager()
    manager.load_profiles()

    while True:
        print("\nVoice Profile Manager Menu:")
        print("1. Generate Random Profile")
        print("2. List Profiles")
        print("3. Save Profiles")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            new_profile = manager.generate_random_profile()
            print(f"Generated new profile: {new_profile.name}")

        elif choice == "2":
            manager.list_profiles()

        elif choice == "3":
            manager.save_profiles()

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number from the menu.")