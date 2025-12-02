from dataclasses import dataclass
from typing import Any
from jinja2 import Template
# Assuming Jinja2 used for templating


@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str | None 
    version: int = 1
    active: bool = False
    
    def update(self, template: str):
        self.template = template
        self.version += 1
        ...
        

    def render(self, parameters: dict[str, Any])->str:
        if self.template is not None:
            template: Template = Template(self.template)
        return template.render(parameters)
