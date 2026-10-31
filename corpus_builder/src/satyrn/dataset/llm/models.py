"""LLM backends: a provider-agnostic Model interface, providers, and a dispatcher."""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jsonschema
from dotenv import load_dotenv
from openai import OpenAI

from satyrn.dataset.llm.context import Context

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SchemaValidationError(Exception):
    pass


class Model(ABC):
    """LLM backend base class."""

    @abstractmethod
    def generate(self, prompt: str, context: Context, thinking: bool = False, effort: str = "medium") -> str | dict:
        """Return the model's response to a prompt rendered with the context.

        Args:
            prompt: The prompt text sent to the model.
            context: Bundles the system prompt, reference documents, and expected output shape.
            thinking: Enable extended thinking.
            effort: One of low, medium, high, xhigh, max.
        """


class DeepSeekModel(Model):
    """A Model backed by DeepSeek's API."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def generate(self, prompt: str, context: Context, thinking: bool = False, effort: str = "medium") -> str | dict:
        messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": context.render(prompt)},
        ]
        create_kwargs = {"model": self.model_name, "messages": messages}
        if thinking:
            create_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            create_kwargs["reasoning_effort"] = effort

        if not context.expect_json:
            response = self.client.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content

        create_kwargs["response_format"] = {"type": "json_object"}
        max_attempts = 3
        for _ in range(max_attempts):
            response = self.client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content
            try:
                parsed = json.loads(content)
                jsonschema.validate(parsed, context.json_schema)
                return parsed
            except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                logger.warning("DeepSeek response did not match the schema: %s", error)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": f"That did not match the required schema: {error}. Try again."}
                )

        raise SchemaValidationError(
            f"DeepSeek did not return a schema-conforming response after {max_attempts} attempts"
        )


@dataclass
class Completion:
    """A model's response, with the details needed to judge it."""

    content: str
    reasoning: str
    finish_reason: str
    completion_tokens: int


class VllmClient:
    """A local vLLM server, called directly to keep each response within the model's own distribution.

    Separate from Model: that interface always adds a system prompt, returns no reasoning or
    finish_reason, and is built around the JSON mode.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_url = os.environ["VLLM_BASE_URL"].rstrip("/")
        self.client = OpenAI(api_key="EMPTY", base_url=f"{self.base_url}/v1")

    def complete(self, prompt: str, thinking: bool, max_tokens: int) -> Completion:
        """Return the model's response to prompt, sent as a single user message."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        choice = response.choices[0]
        # The field's name depends on the vLLM version; the OpenAI client keeps unknown fields as attributes.
        reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None)
        return Completion(
            content=choice.message.content or "",
            reasoning=reasoning or "",
            finish_reason=choice.finish_reason,
            completion_tokens=response.usage.completion_tokens,
        )

    def count_tokens(self, text: str) -> int:
        """Return the prompt token count of a user message holding text, via vLLM's /tokenize."""
        # An absolute URL, because /tokenize is outside the /v1 prefix.
        response = self.client.post(
            f"{self.base_url}/tokenize",
            cast_to=object,
            body={"model": self.model_name, "messages": [{"role": "user", "content": text}]},
        )
        return response["count"]


def get_llm(provider: str, model_name: str) -> Model:
    """Return a Model for provider backed by model_name."""
    if provider == "deepseek":
        return DeepSeekModel(model_name)
    raise ValueError(f"Unknown provider: {provider}")
