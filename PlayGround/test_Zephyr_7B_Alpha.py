from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification, BitsAndBytesConfig, pipeline
import torch
from rapidfuzz import process
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList

class test_Zephyr_7B_Alpha:

    def __init__(self):
        # Load the Zephyr-7B-Alpha model
        zephyr_model_name = "HuggingFaceH4/zephyr-7b-alpha"

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Initialize Zephyr model

        self.zephyr_tokenizer = AutoTokenizer.from_pretrained(zephyr_model_name)
        self.zephyr_model = AutoModelForCausalLM.from_pretrained(zephyr_model_name, quantization_config=quantization_config)
        self.zephyr_model.generation_config.pad_token_id = self.zephyr_tokenizer.pad_token_id

        self.zephyr_pipe = pipeline("text-generation", model=self.zephyr_model, tokenizer=self.zephyr_tokenizer, device_map="auto")


    def generate_response_using_zephyr(self, movie_info: dict, user_query: str) -> str:
            """
            Generate a response using Zephyr-7B-Alpha based on the user's intent and the query result.
            """
            system_msg = '''
            You are a specialized movie chatbot. 

            <Requirements>
            INPORTANT: Provide only 1 - 2 sentence as response, as short as possible
            INPORTANT: Maximum 100 characters or 50 words
            INPORTANT: Do not return JSON format
            Do not return Requirements in your response
            <Requirements>

            Prioritize the provided information to formulate your response. 
            Use your own knowledge about movies if you think <data> part does not provide enough knowledge
            If <User query> is not related to general movie topic, gently remind user to focus on movie-related topics.
            '''

            prompt_info = f"<|system|>: \"{system_msg}\"\n"
            prompt_info += f"<|data|>: \"{movie_info}\"\n"

            # # Format the prompt with the movie information and user query
            # prompt = prompt_info

            # # Tokenize and prepare the input for the model
            # inputs = self.zephyr_tokenizer(prompt, return_tensors="pt").to("cuda")

            # output = self.zephyr_model.generate(inputs).to("cuda")

            # response = self.zephyr_tokenizer.decode(output[0], skip_special_tokens=True)
            # print(response)
            


            # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
            messages = [
                {
                    "role": "system",
                    "content": f"{prompt_info}",
                },
                {"role": "user", "content": f"{user_query}"},
            ]
            prompt = self.zephyr_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if 'quantization_config' in prompt:
                del prompt['quantization_config']

            outputs = self.zephyr_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            
            response = outputs[0]["generated_text"]
            print(response)

            return response
