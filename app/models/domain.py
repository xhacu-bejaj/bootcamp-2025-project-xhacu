from dataclasses import dataclass, field
from jinja2 import Environment, Template, UndefinedError, select_autoescape




PROMPT_ENV = Environment(
    trim_blocks=True, 
    lstrip_blocks=True,
    autoescape=select_autoescape(['html', 'xml']) 
)

# TODO: create an experiment.py and see how jinja works
@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str | None 
    version: int = 1
    active: bool = False
    
    _compiled_template: Template | None = field(init=False, repr=False)
    
    def __post_init__(self):
        if self.template is None:
            self._compiled_template = None
            return

        self._compiled_template = PROMPT_ENV.from_string(self.template)


    def update(self, template: str):
        # When updating, we must re-compile the template
        self.template = template
        self._compiled_template = PROMPT_ENV.from_string(template) # Compile new string
        self.version += 1


    def render(self, **kwargs: str) -> str:
        if self._compiled_template is None:
            raise ValueError("Cannot render prompt: template is None.")
        try:
            # pre-compiled template object to render
            return self._compiled_template.render(**kwargs)
            
        except UndefinedError as e:
            raise KeyError(f"Missing required variable in prompt rendering: {e}")