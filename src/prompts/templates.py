"""Prompt template management — load from YAML, format with context."""

from typing import Dict, Any, Optional
import re


class PromptTemplate:
    """A prompt template using {variable} placeholders."""

    def __init__(self, name: str, template: str, metadata: Optional[Dict] = None):
        self.name = name
        self.template = template
        self.metadata = metadata or {}

    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing variable {e} in template '{self.name}'. "
                f"Required: {self.get_variables()}"
            )

    def get_variables(self) -> list:
        return re.findall(r"\{(\w+)\}", self.template)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PromptTemplate":
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            template=data["template"],
            metadata={k: v for k, v in data.items() if k not in ("name", "template")},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "variables": self.get_variables(), **self.metadata}

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', vars={self.get_variables()})"
