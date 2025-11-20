from dataclasses import dataclass

@dataclass
class Prompt:
    id: str
    purpose: str
    name: str
    template: str
    version: int = 1
    #active:bool = False # added by me, default is false because user could create Prompt just to store
                        # them and activate in another time---------- Instead of this flag, consider a Prompt active if is inside
                        # the self._active_prompts in app/services/prompt_store.py

    def update(self, template: str):
        ...

    def render(self, **kwargs: str):
        ...
