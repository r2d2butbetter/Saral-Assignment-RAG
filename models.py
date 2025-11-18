from pydantic import BaseModel, Field


class ScriptDelta(BaseModel):
    """Data model for a single change to a script."""
    reason_for_change: str = Field(description="Why this change was made, citing the user request")
    old_text_segment: str = Field(description="The exact, original text segment that is being changed.")
    new_text_segment: str = Field(description="The new, updated text segment.")


class FullScriptUpdate(BaseModel):
    """Data model for the full updated script and the list of changes."""
    changes: list[ScriptDelta]
    full_updated_script: str = Field(description="The complete, new script with all changes applied")
