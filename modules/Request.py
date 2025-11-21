import requests
from langchain.llms.base import LLM

class RemoteLLM(LLM):
    api_url: str
    api_key: str

    def _call(self, prompt, stop=None):
        res = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"prompt": prompt, "max_length": 300}
        )
        try:
            return res.json().get("response", "")
        except:
            return f"[ERROR]: {res.text}"

    @property
    def _llm_type(self):
        return "remote"
