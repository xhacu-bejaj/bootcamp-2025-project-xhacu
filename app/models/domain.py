from dataclasses import dataclass

@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str|None
    version: int = 1
    active:bool = False

    def update(self, template: str):
        ...

    def render(self, **kwargs: str):
        ...
