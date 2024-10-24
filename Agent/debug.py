from Agent.response_generator import response_generator

class debug:

    def __init__(self):
        # Initialize the SPARQL executor with the dataset path
        self.response_generator = response_generator()

    def get_response(self, message: str) -> str:
        """
        Generate a response to a input message.

        Args:
            message (str): The input message.

        Returns:
            str: The response to the input message.
        """
        return self.response_generator.get_response(message)

if __name__ == '__main__':
    debugBot = debug()
    while True:
        user_input = input("Enter your message (or 'exit' to exit): ")
        if user_input.lower() == 'exit':
            break
        response = debugBot.get_response(user_input)
        print("Response: ", response)