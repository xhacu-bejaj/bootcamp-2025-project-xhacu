from typing import Optional

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column
from sqlalchemy import Text

import uuid


class Prompt(SQLModel, table=True):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), primary_key=True
    )
    purpose: str
    name: str = Field(index=True)
    template: str | None = None

    version: int = Field(default=1)
    active: bool = Field(default=False)

    usages: list["PromptUsage"] = Relationship(back_populates="prompt")


class PromptUsage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    prompt_id: str = Field(foreign_key="prompt.id", index=True)
    prompt: Prompt = Relationship(back_populates="usages")

    document_text: str = Field(sa_column=Column(Text, nullable=False))
    llm_response: str = Field(sa_column=Column(Text, nullable=False))
    latency: int

    model_used: str
