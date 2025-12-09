from dataclasses import dataclass, asdict, field
from jinja2 import Template


@dataclass
class Chunk:
    """Represents a chunk of text stored in the vector database."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


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

    def render(self, parameters: dict) -> str:
        """Render the prompt template with given parameters using Jinja2.
        
        Args:
            parameters: Dictionary of parameters to render the template with
            
        Returns:
            Rendered template string
        """
        if self.template is not None:
            template = Template(self.template)
            return template.render(parameters)
        return ""
