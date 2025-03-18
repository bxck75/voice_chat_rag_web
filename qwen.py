'''
## reference
- https://github.com/abetlen/llama-cpp-python/blob/main/examples/gradio_chat/local.py
- https://github.com/awinml/llama-cpp-python-bindings
- https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py
- https://github.com/abetlen/llama-cpp-python/blob/main/examples/gradio_chat/server.py
- https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/model.py
- https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py'''


import json
import copy
import os
import psutil
import llama_cpp
from transformers import AutoTokenizer

from models.base_model import Simulator
from utils.logging_util import logger
import config


class Qwen2Simulator(Simulator):

    def __init__(self, system_list=None):
        local_path = "/workspace/xusong/huggingface/models/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
        if os.path.exists(local_path):
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                "/workspace/xusong/huggingface/models/Qwen2-0.5B-Instruct/")
            self.llm = llama_cpp.Llama(  # n_ctx, n_threads
                model_path=local_path,
                # 默认的tokenizer有bug，tokenize后的id不同
                tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer(self.hf_tokenizer),
                n_ctx=config.MAX_SEQUENCE_LENGTH,  #
                # n_threads=None, # 默认会根据cpu数来设置 n_threads
                # use_mlock=True,
                verbose=True,
            )
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
            self.llm = llama_cpp.Llama.from_pretrained(
                repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
                tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer(self.hf_tokenizer),
                filename="*fp16.gguf",
                n_ctx=config.MAX_SEQUENCE_LENGTH,
                # use_mlock=True,
                verbose=True,
            )
        logger.info(f"llm has been initialized: {self.llm}, "
                    f"n_threads={self.llm.n_threads}, n_ctx={self.llm.n_ctx}, "
                    f"env[CACHE]={os.environ.get('CACHE', None)}")


        # qwen2-0.5b-chat 有时内容生成结束没有<|im_end|>，直接跟 <|im_start|>
        self.assistant_stop_words = [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
        ]
        self.assistant_stop_tokens = self.tokenize("".join(self.assistant_stop_words))
        self.user_stop_words = self.assistant_stop_words + ["？", "?"]
        self.user_stop_tokens = self.tokenize("".join(self.user_stop_words))
        logger.info(f"assistant_stop_tokens: {self.assistant_stop_tokens}")
        logger.info(f"user_stop_tokens: {self.user_stop_tokens}")

        self.generation_kwargs = dict(
            temperature=config.DEFAULT_TEMPERATURE,
            top_p=config.DEFAULT_TOP_P,
            top_k=config.DEFAULT_TOP_K,
            max_tokens=config.DEFAULT_MAX_NEW_TOKENS,
            repeat_penalty=1.1,
        )
        self.user_start_tokens = self.tokenize("<|im_start|>user\n")
        self.assistant_start_tokens = self.tokenize("<|im_start|>assistant\n")
        # self.llm.generate  .set_cache   .last_n_tokens_size  .reset  .ctx ._ctx

        # cache = llama_cpp.LlamaDiskCache(capacity_bytes=cache_size)
        cache = llama_cpp.LlamaRAMCache(capacity_bytes=2 << 30)  # 2G
        self.llm.set_cache(cache)

        if system_list is not None:
            self.pre_cache_system(system_list)

    def tokenize(self, text):
        return self.llm.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens):
        return self.llm.detokenize(tokens).decode("utf-8")

    def strip_stoptokens(self, tokens):
        while tokens and tokens[0] in self.assistant_stop_tokens:
            logger.info(f"head-striping {tokens[0]} {self.detokenize([tokens[0]])}")
            tokens.pop(0)
        while tokens and tokens[-1] in self.assistant_stop_tokens:
            logger.info(f"tail-striping {tokens[-1]} {self.detokenize([tokens[-1]])}")
            tokens.pop()
        return tokens

    def generate(self, history, stream=True):
        """
        额外前向：remains 5 to forward "<|im_end|>\n<|im_start|>assistant\n"
        :param history:
        :param stream:
        :return:
        """
        if history[-1]['role'] in ["user"]:
            start_tokens = self.assistant_start_tokens
            stop_words = self.assistant_stop_words
            suffix_tokens = self.user_start_tokens

        elif history[-1]['role'] in ["assistant", "system"]:
            start_tokens = self.user_start_tokens
            stop_words = self.user_stop_words
            suffix_tokens = self.assistant_start_tokens

        input_ids = []
        for message in history:
            if "tokens" not in message:  # tokens
                message["tokens"] = self.tokenize(message["content"])
            input_ids += self.tokenize(f"<|im_start|>{message['role']}\n") \
                         + message["tokens"] \
                         + self.tokenize("<|im_end|>\n")
        input_ids += start_tokens
        if stream:
            return self._stream_generate(input_ids, stop_words, suffix_tokens)
        else:
            return self._generate(input_ids)

    def _stream_generate(self, input_ids, stop_words, suffix_tokens=None):
        logger.info(f"generation_kwargs {self.generation_kwargs}")
        output = self.llm.create_completion(
            input_ids,
            stream=True,
            stop=stop_words,
            **self.generation_kwargs
        )
        # TODO: 检测finish reason，如果是length，则shift，并继续生成。
        # TODO: 返回 token_id,
        for out in output:
            stream = copy.deepcopy(out)
            if stream["choices"][0]["finish_reason"] is None:
                yield stream["choices"][0]["completion_text"], stream["choices"][0]["completion_tokens"]
            else:
                logger.info(
                    f'finish_reason {stream["choices"][0]["finish_reason"]} with text: {stream["choices"][0]["text"]}')

        #
        self.post_cache(suffix_tokens)

    def pre_cache_system(self, system_list):
        """ warmup for system prompt
        :param system_list:
        :return:
        """
        logger.info(f"cache size {self.llm.cache.cache_size}")
        for system_prompt in system_list:
            logger.info(f"pre caching '{system_prompt}'")
            input_ids = self.tokenize(f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n")
            _output = self.llm.create_completion(
                input_ids,
                stream=False,
                max_tokens=1,
                top_k=1
            )
            logger.info(
                f"cache size {self.llm.cache.cache_size}={self.llm.cache.cache_size / 1024 / 1024 / 1024:.2f} GB, "
                f"process_mem: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

        self._disable_cache()

    def post_cache(self, suffix_tokens):
        """ warmup for next turn generation
        :param suffix_tokens:
        :return:
        """
        logger.info(f"cache size {self.llm.cache.cache_size}={self.llm.cache.cache_size / 1024 / 1024 / 1024:.2f} GB, "
                    f"process_mem: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        if suffix_tokens:
            logger.info(f"before warmup: n_tokens = {self.llm.n_tokens}")
            self.llm.eval([151645, 198] + suffix_tokens)  # <|im_end|>\n
            logger.info(f"after warmup: n_tokens = {self.llm.n_tokens}")
        logger.info(f"cache size {self.llm.cache.cache_size}={self.llm.cache.cache_size / 1024 / 1024 / 1024:.2f} GB, "
                    f"process_mem: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

    def _disable_cache(self):
        llama_cpp.LlamaRAMCache.__setitem__ = lambda *args: None
        llama_cpp.Llama.save_state = lambda *args: None


if __name__ == "__main__":

    bot = Qwen2Simulator()
    messages = [{"role": "system", "content": "你是一个导游。"}]
    generated_tokens = None
    print("######## requesting", messages)
    for generated_text, generated_tokens in bot.generate(messages, stream=True):
        print(generated_text, generated_tokens)

    for i in range(3):
        generated_tokens = bot.strip_stoptokens(generated_tokens)
        messages.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": generated_text, "tokens": generated_tokens})
        print("######## requesting", messages)
        for generated_text, generated_tokens in bot.generate(messages, stream=True):
            pass
            # print(generated_text, all_tokens)
