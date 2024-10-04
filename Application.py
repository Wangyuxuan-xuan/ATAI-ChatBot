from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from sparql_processor import SPARQLProcessor
SPEAK_EASY_HOST = 'https://speakeasy.ifi.uzh.ch'

SPEAK_EASY_USERNAME = "yuxuan.wang"
SPEAK_EASY_PASSWORD = "D2rdC6Q9"
BOT_NAME = "red-dragon Bot"
BOT_PASSWORD = "X0ynU0H9"

LISTEN_FREQENCY = 2
GRAPH_PATH = './Dataset/14_graph.nt'

class Agent:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        
        # Initialize KG
        self.sparql_processor = SPARQLProcessor(GRAPH_PATH)

        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=SPEAK_EASY_HOST, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=False, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #
                    response = self.sparql_processor.generate_response(message.message)
                    
                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    
                    print(f"The response is: {response}")
                    
                    # room.post_messages(response)
                    
                    # room.post_messages(f"I Received your message: '{message.message}' ")
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(LISTEN_FREQENCY)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent(SPEAK_EASY_USERNAME, SPEAK_EASY_PASSWORD)
    demo_bot.listen()
