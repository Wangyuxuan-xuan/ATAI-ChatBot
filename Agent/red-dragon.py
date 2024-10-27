from speakeasypy import Speakeasy, Chatroom
from typing import List
import time

# ADDED BY OMER
from Agent.response_generator import response_generator

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:

    def __init__(self, username, password):
        # Initialize the SPARQL executor with the dataset path
        self.response_generator = response_generator()

        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        
        while True:
            
            # 'rooms' is the list of active chatrooms.
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            
            # Iterate each active chatroom
            for room in rooms:

                # If the room is not initiated by the agent
                if not room.initiated:

                    # Send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
 
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    
                    # FOR INFORMING DEVELOPER ONLY
                    print(
                        f"- <new message> #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # ******************** 
                    # Implement your agent here #

                    response = self.get_response(message.message, room)
                    # print the response
                    print(
                        f"- <response> #{message.ordinal}: '{response}' "
                        f"- {self.get_time()}")
                    
                    response = response.encode('utf-8')
                    room.post_messages(response.decode('latin-1'))
                    
                    # room.post_messages(response)
                    
    
                    # ********************
                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    # room.post_messages(f"Received your message: '{message.message}' ")
                    
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                """
                # NOW, LET'S NOT THINK ABOUT THE REACTIONS! WE SHOULD FOCUS ON THE SPARQL QUERYING.

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

                """

            time.sleep(listen_freq)


    def get_response(self, message: str, room) -> str:
        """
        Generate a response to a input message.

        Args:
            message (str): The input message.
            room: The chatroom object.

        Returns:
            str: The response to the input message.
        """
        
        try:
            self.response_generator.set_room(room)
            response = self.response_generator.get_response(message)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            response = "I apologize, but I encountered an error while processing your request. Please try again :("

        return response
        

    
    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("red-dragon", "X0ynU0H9")
    demo_bot.listen()

    
