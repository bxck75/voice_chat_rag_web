import os,requests,io,warnings,torch,speech_recognition
from hugchat import hugchat
from hugchat.login import Login
from typing import Union, List, Generator
from requests.sessions import RequestsCookieJar
from dotenv import load_dotenv,find_dotenv
from hugchat.message import Message
from hugchat.types.assistant import Assistant
from hugchat.types.model import Model
from hugchat.types.message import MessageNode, Conversation
from new_voice_document_processor import DocumentProcessor
from typing import Any, List, Mapping, Optional
from system_prompts import __all__ as prompts
import speech_recognition
from TTS.api import TTS
from rich import print as rp
import time
from playsound import playsound

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")


class EnhancedChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/',default_llm=1):
        self.prompts=prompts
        self.email = email
        self.password = password
        self.default_llm = default_llm
        self.cookie_path_dir = cookie_path_dir
        self.system_prompt = self.prompts[0] # default_system_prompt
        self.cookies = self.login()
        self.bot = hugchat.ChatBot(cookies=self.cookies.get_dict(), 
                                       default_llm = self.default_llm
                                       )
        self.conv_id = None
        

    def login(self):
        print("Attempting to log in...")
        sign = Login(self.email, self.password)
        try:
            cookies = sign.login(cookie_dir_path=self.cookie_path_dir, save_cookies=True)
            print("Login successful!")
            return cookies
        except Exception as e:
            print(f"Login failed: {e}")
            print("Attempting manual login with requests...")
            self.manual_login()
            raise

    def manual_login(self):
        login_url = "https://huggingface.co/login"
        session = requests.Session()
        response = session.get(login_url)
        print("Response Cookies:", response.cookies)
        print("Response Content:", response.content.decode())
        
        csrf_token = response.cookies.get('csrf_token')
        if not csrf_token:
            print("CSRF token not found in cookies.")
            return
        
        login_data = {
            'email': self.email,
            'password': self.password,
            'csrf_token': csrf_token
        }
        
        response = session.post(login_url, data=login_data)
        if response.ok:
            print("Manual login successful!")
        else:
            print("Manual login failed!")


    def document_processor(self, repo_url=None, document=None, delete=False):
        self.document_ids = {} 
        if not hasattr(self, 'processor'):
            self.processor = DocumentProcessor(llm=self.bot)

        if repo_url:
            documents = self.processor.load_documents_from_github(repo_url)
            split_docs = self.processor.split_documents(documents)
            embeddings = self.processor.embed_documents(split_docs)
            self.processor.create_vectorstore(split_docs, embeddings)
            self.processor.setup_retriever()

        if delete == True and document:
            self.processor.delete_document(document)


    def __call__(self, text): 
        if self.conv_id:
            self.bot.change_conversation(self.bot.get_conversation_from_id(self.conv_id))
        else:
            self.conv_id = self.bot.new_conversation(system_prompt=self.system_prompt, modelIndex=self.default_llm, switch_to=True)
        return self.send_message(text)

    def send_message(self, message,web=False):
        message_result = self.bot.chat(message,web_search=web)
        return message_result.wait_until_done()

    def stream_response(self, message, web=False,stream=False):
        responses=[]
        for resp in self.bot.query(message, stream=stream,web_search=web):
            #rp(resp['token'])
            responses.append(resp['token'])
        return ' '.join(responses)
    
    def web_search(self, text):
        result = self.send_message(text, web=True)
        return result

    def retrieve_context(self, query: str):
        results = self.processor.retrieve_similar_documents(query)
        context = "\n".join([doc.page_content for doc in results])
        return context
    
    def delete_all_conversations(self):
        self.bot.delete_all_conversations()

    def delete_conversation(self, conversation_object: Conversation = None):
        self.bot.delete_conversation(conversation_object)

    def get_available_llm_models(self) -> list:
        return self.bot.get_available_llm_models()

    def get_remote_conversations(self, replace_conversation_list=True):
        return self.bot.get_remote_conversations(replace_conversation_list)

    def get_conversation_info(self, conversation: Union[Conversation, str] = None) -> Conversation:
        return self.bot.get_conversation_info(conversation)

    def get_assistant_list_by_page(self, page: int) -> List[Assistant]:
        return self.bot.get_assistant_list_by_page(page)

    def search_assistant(self, assistant_name: str = None, assistant_id: str = None) -> Assistant:
        return self.bot.search_assistant(assistant_name, assistant_id)

    def switch_model(self, index):
        self.conv_id=None
        self.default_llm=index

    def switch_conversation(self, id):
        self.conv_id = id

    def switch_role(self,system_prompt_id):
        self.system_prompt=self.prompts[system_prompt_id]
        #return self.bot.new_conversation(switch_to=True, system_prompt=self.system_prompt)

    def chat(
        self,
        text: str,
        web_search: bool = False,
        _stream_yield_all: bool = False,
        retry_count: int = 5,
        conversation: Conversation = None,
        *args,
        **kwargs
    ) -> Message:
        return self.bot.chat(text, web_search, _stream_yield_all, retry_count, conversation, *args, **kwargs)


    @staticmethod
    def listen_for_speech():
        recognizer = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except speech_recognition.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except speech_recognition.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    @staticmethod
    def optimized_tts(
        text: str,
        output_file: str = "output.wav",
        model_name: str = "tts_models/en/ljspeech/fast_pitch",
        speaking_rate: float = 1.9
    ) -> str:
        start_time = time.time()

        tts = TTS(model_name=model_name)

        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker=tts.speakers[0] if tts.speakers else None,
            language=tts.languages[0] if tts.languages else None,
            speed=speaking_rate
        )

        end_time = time.time()
        print(f"TTS generation took {end_time - start_time:.2f} seconds")
        return output_file

    @staticmethod
    def play_mp3(file_path):
        playsound(file_path)

    def continuous_voice_chat(self, story_teller_prompt="You are a helpful assistant."):
        self.input_method=None
        while True:
            if self.input_method == None:
                print("Speak your query (or say 'exit' to quit):")
                self.input_method=self.listen_for_speech()
            
            # we start in voice mode
            query = self.input_method
            
            if query is None:
                continue

            if query.lower() == "voice":
                print("Speak your query (or say 'exit' to quit):")
                self.input_method = self.listen_for_speech()
                

            if query.lower() == "type":
                self.input_method = input("Type your question(or type 'exit' to quit): \n")
                

            if 'switch prompt ' in query.lower():
                q = query.lower()
                new_prompt = q.split("switch prompt ").pop().replace(" ", "_")
                # try fetch the prompt key by name value
                try:
                    new_prompt_id = prompts.index(new_prompt)
                    rp(f"new prompt key:{new_prompt_id}")
                except ValueError:
                    rp(f"{new_prompt} not found in prompts list! set to key 0!(default_system_prompt)")
                    new_prompt_id = 0
                
                self.switch_role(new_prompt_id)
                query=''
                self.optimized_tts(f"Switched Role to {new_prompt}!")
                self.play_mp3('output.wav')
                
                

            if query.lower() == 'exit':
                rp("Goodbye!")
                self.optimized_tts("Ok, exiting!")
                self.play_mp3('output.wav')
                break
            
            # search for content
            result = self.web_search(query)
            # check empty results
            web_context = "/n".join(result) if not result == '' else None
            # Retrieve context from documents
            vector_context = self.retrieve_context(query)
            # lace the system prompt
            self.system_prompt.replace("<<VSCONTEXT>>", vector_context) if vector_context else self.system_prompt.replace("<<VSCONTEXT>>", "No Context Available in the vectorstore!")
            self.system_prompt.replace("<<WSCONTEXT>>", web_context) if web_context else self.system_prompt.replace("<<WSCONTEXT>>", "No Context Available from the websearch!")
            # chat
            response = self.bot.chat(query)
            # check if bot wants to store knowledge
            if "/Store:" in response:
                url=response.split("/Store:").pop().split(" ")[0]
                rp(f"Fetching and storing data from link: {url}")
                try:
                    self.document_processor(url)
                except Exception as e:
                    rp(f"Error while fetching data from {url}! {e}")
                continue
            # check if bot wants to delete knowledge
            if "/Delete:" in response:
                document=response.split("/Delete:").pop().split(" ")[0]
                rp(f"Deleting {document} from vectorstore!")
                try:
                    self.init_document_processor(document, delete=True)
                except Exception as e:
                    rp(f"Error while deleting {document} from vectorstore! {e}")

            rp(f"Chatbot: {response}")
            # to voice
            self.optimized_tts(str(response))
            self.play_mp3('output.wav')
    
# Example usage
if __name__ == "__main__":

    bot = EnhancedChatBot(os.getenv("EMAIL"), os.getenv("PASSWD"),default_llm=1)
    print("Starting in voice mode. Speak into your microphone.(say'type' to enter text input mode.)")
    bot.continuous_voice_chat()