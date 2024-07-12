class VoiceProfile:
    def __init__(self, name, language, pitch, speaking_rate):
        self.name = name
        self.language = language
        self.pitch = pitch
        self.speaking_rate = speaking_rate

    def to_dict(self):
        return {
            'name': self.name,
            'language': self.language,
            'pitch': self.pitch,
            'speaking_rate': self.speaking_rate
        }

    @classmethod
    def from_dict(cls, profile_dict):
        return cls(
            profile_dict['name'],
            profile_dict['language'],
            profile_dict['pitch'],
            profile_dict['speaking_rate']
        )