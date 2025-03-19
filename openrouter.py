import os
import json
import requests
import time
from typing import List, Union, Generator, Dict, Iterator
from pydantic import BaseModel, Field

DEBUG = True


class Pipe:
    class Valves(BaseModel):
        """Configuration for OpenRouter and User Credit API."""
        FREE_ONLY: bool = Field(default=False)
        USER_CREDIT_CHECK_URL: str = Field(default="")
        USER_CREDIT_API_KEY: str = Field(default="")
        OPENROUTER_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "openrouter"
        self.name = "openrouter/"
        self.OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


        self.valves = self.Valves(
            **{
                "FREE_ONLY": os.getenv("FREE_ONLY", "false").lower() == "true",
                "USER_CREDIT_CHECK_URL": os.getenv("USER_CREDIT_CHECK_URL", ""),
                "USER_CREDIT_API_KEY": os.getenv("USER_CREDIT_API_KEY", ""),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            }
        )
        if not self.valves.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        self.pipelines = []

    def _debug(self, message: str):
        if DEBUG:
            print(message)

    def _get_headers(self, extra_headers: dict = None) -> Dict[str, str]:
        """Gets headers for OpenRouter API requests."""
        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _handle_response(self, response: requests.Response) -> dict:
        """Handles API responses, raising exceptions for errors."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self._debug(f"HTTPError: {e.response.text}")
            raise
        except ValueError as e:
            self._debug(f"Invalid JSON response: {response.text}")
            raise

    def _format_model_id(self, model_id: str) -> str:
        """Formats the model ID for OpenRouter."""
        if model_id.startswith("openrouter."):
            model_id = model_id[len("openrouter.") :]
        elif model_id.startswith("openroutermodels."):
            model_id = model_id[len("openroutermodels.") :]
        return model_id

    def get_openrouter_models(self) -> List[Dict[str, str]]:
        """Gets the list of available models from OpenRouter."""
        url = f"{self.OPENROUTER_API_BASE_URL}/models"
        try:
            self._debug(f"Fetching models from {url}")
            response = requests.get(url, headers=self._get_headers())
            models_data = self._handle_response(response).get("data", [])
            if self.valves.FREE_ONLY:
                models_data = [
                    model for model in models_data if "free" in model.get("id", "").lower()
                ]
            return [
                {
                    "id": model.get("id", "unknown"),
                    "name": model.get("name", "Unknown Model"),
                    "pricing_completion": model.get("pricing", {}).get("completion", 0.0),
                    "pricing_prompt": model.get("pricing", {}).get("prompt", 0.0)
                }
                for model in models_data
            ]
        except Exception as e:
            self._debug(f"Failed to fetch models: {e}")
            return [{"id": "openrouter", "name": str(e)}]

    def pipes(self) -> List[dict]:
        """Returns the list of available models (required by Open WebUI)."""
        return self.get_openrouter_models()


    def _check_user_credit(self, user_id: str) -> float:
        """Checks user credit via your website's API."""
        if not self.valves.USER_CREDIT_CHECK_URL:
            self._debug("USER_CREDIT_CHECK_URL not set, skipping credit check.")
            return 1.0

        headers = {"Authorization": f"Bearer {self.valves.USER_CREDIT_API_KEY}"}
        try:
            response = requests.get(
                f"{self.valves.USER_CREDIT_CHECK_URL}/{user_id}",
                headers=headers,
            )
            response.raise_for_status()
            credit_data = response.json()
            return float(credit_data.get("credit", 0.0))
        except requests.RequestException as e:
            self._debug(f"Credit check failed: {e}")
            return 0.0

    def _get_model_cost(self, model_id: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
        """Calculates the cost of a request based on model pricing."""
        models = self.get_openrouter_models()
        for model in models:
            if model["id"] == model_id:
                prompt_cost = model.get("pricing_prompt", 0.0) * prompt_tokens
                completion_cost = model.get("pricing_completion", 0.0) * completion_tokens
                return prompt_cost + completion_cost
        self._debug(f"Model cost not found for: {model_id}")
        return 0.0


    def _deduct_credit(self, user_id: str, cost: float):
        """Deducts credit via your website's API."""
        if not self.valves.USER_CREDIT_CHECK_URL:
            self._debug("USER_CREDIT_CHECK_URL not set, skipping credit deduction.")
            return

        headers = {"Authorization": f"Bearer {self.valves.USER_CREDIT_API_KEY}"}
        payload = {"user_id": user_id, "cost": cost}

        try:
            response = requests.post(
                f"{self.valves.USER_CREDIT_CHECK_URL}/deduct",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            self._debug(f"Deducted {cost} from user {user_id}")
        except requests.RequestException as e:
            self._debug(f"Credit deduction failed: {e}")



    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator[str, None, None]]:
        """Main pipeline function to handle chat requests."""
        try:
            model = self._format_model_id(model_id)
            stream = body.get("stream", False)
            user_id = body.get("user", "")

            if not user_id:
                return "Error: user_id is required"

            if not self.valves.FREE_ONLY:
                user_credit = self._check_user_credit(user_id)
                if user_credit <= 0:
                    return "Error: Insufficient credit"

            if DEBUG:
                self._debug("Incoming body:")
                self._debug(json.dumps(body, indent=2))
                self._debug(f"Using model: {model}")

            if stream:
                return self.stream_response(model, messages, user_id)
            else:
                return self.get_completion(model, messages, user_id)

        except KeyError as e:
            error_msg = f"Missing required key in body: {e}"
            self._debug(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            self._debug(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(
        self, model: str, messages: List[dict], user_id: str, retries: int = 5
    ) -> Generator[str, None, None]:
        """Handles streaming responses from OpenRouter."""
        url = f"{self.OPENROUTER_API_BASE_URL}/chat/completions"
        payload = {"model": model, "messages": messages, "stream": True}
        self._debug(f"Streaming response from {url}")
        self._debug(f"Payload: {json.dumps(payload, indent=2)}")

        total_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        for attempt in range(retries):
            try:
                response = requests.post(
                    url, json=payload, headers=self._get_headers(), stream=True
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            line_data = line.decode("utf-8").lstrip("data: ")
                            if line_data == "[DONE]":
                                if not self.valves.FREE_ONLY:
                                    self._deduct_credit(user_id, total_cost)
                                break
                            event = json.loads(line_data)
                            delta_content = (
                                event.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if delta_content:
                                completion_tokens += 1
                                yield delta_content

                        except json.JSONDecodeError:
                            self._debug(f"Failed to decode stream line: {line}")
                            continue


                if prompt_tokens == 0:
                    prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)
                    total_cost += self._get_model_cost(model, prompt_tokens=prompt_tokens)

                total_cost += self._get_model_cost(model, completion_tokens=completion_tokens)
                completion_tokens = 0
                return


            except requests.RequestException as e:
                 if response and response.status_code == 429 and attempt < retries - 1:
                    wait_time = 2**attempt
                    self._debug(f"Rate limited. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                 else:
                    self._debug(f"Stream request failed: {e}")
                    yield f"Error: {str(e)}"
                    return

    def get_completion(self, model: str, messages: List[dict], user_id: str, retries: int = 3) -> str:
        """Handles non-streaming (single response) requests."""
        url = f"{self.OPENROUTER_API_BASE_URL}/chat/completions"
        payload = {"model": model, "messages": messages}
        self._debug(f"Sending completion request to {url}")
        self._debug(f"Payload: {json.dumps(payload, indent=2)}")

        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, headers=self._get_headers())
                data = self._handle_response(response)
                completion_content = data["choices"][0]["message"]["content"]


                prompt_tokens = sum(len(msg.get("content","").split()) for msg in messages)
                completion_tokens = len(completion_content.split())

                total_cost = self._get_model_cost(model, prompt_tokens, completion_tokens)

                if not self.valves.FREE_ONLY:
                    self._deduct_credit(user_id, total_cost)

                return completion_content

            except requests.RequestException as e:
                if response and response.status_code == 429 and attempt < retries - 1:
                    wait_time = 2**attempt
                    self._debug(f"Rate limited. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._debug(f"Completion request failed: {e}")
                    return f"Error: {str(e)}"
