import backoff 
import openai
import os
import asyncio
from typing import Any
import sys

# --- Define a handler to print when we hit a rate limit ---
def backoff_hdlr(details):
    exc = details.get("exception")
    print("\n[Backoff] Exception during OpenAI call:")
    print("  type   :", type(exc).__name__)
    print("  message:", exc)
    print(f"  sleeping for {details['wait']:.1f}s... (Try #{details['tries']})")

# Add the handler to the decorators
@backoff.on_exception(backoff.expo, openai.error.RateLimitError, on_backoff=backoff_hdlr)
@backoff.on_exception(backoff.expo, openai.error.APIConnectionError, on_backoff=backoff_hdlr)
@backoff.on_exception(backoff.expo, openai.error.ServiceUnavailableError, on_backoff=backoff_hdlr)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, on_backoff=backoff_hdlr)
@backoff.on_exception(backoff.expo, openai.error.APIConnectionError, on_backoff=backoff_hdlr)
@backoff.on_exception(backoff.expo, openai.error.ServiceUnavailableError, on_backoff=backoff_hdlr)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# ... (The rest of the file remains the same) ...

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
    
    # ... (Keep the rest of batch_generate etc if needed) ...