from dataclasses import dataclass, field



# TODO: create an experiment.py and see how jinja works
@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str | None 
    version: int = 1
    active: bool = False
    
    def update(self, template: str):
        # self.template = template
        # self.version += 1
        ...
        

    def render(self, **kwargs: str):
        ...
