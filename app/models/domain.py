from dataclasses import dataclass, asdict
from typing import Any
from jinja2 import Template


@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str | None
    version: int = 1
    active: bool = False

    def update(self, template: str):
        """Update the prompt template and increment version.
        
        Args:
            template: New template string
        """
        self.template = template
        self.version += 1

    def model_dump(self) -> dict:
        """Convert dataclass to dictionary for Pydantic serialization."""
        return asdict(self)

    def render(self, parameters: dict[str, Any]) -> str:
        if self.template is not None:
            template: Template = Template(self.template)
        return template.render(parameters)
