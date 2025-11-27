from dataclasses import dataclass

@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str|None
    version: int = 1
    active:bool = False

    # needed for predict endpoint
    def update(self, template: str):
        self.template = template

    def render(self, **kwargs: str):
        ...
