from pydantic import BaseModel, Field
from typing import List, Optional,Literal, Union

class SummaryAgentOutputModel(BaseModel):
    kind: Literal["SUMMARY"] = "SUMMARY"
    task: Literal["Cover Letter", "CV"] = Field(
        ...,
        description="Either 'Cover Letter' or 'CV' depending on what the user asks for in the query"
    )
    summary: str
class SemanticAlignmentAgentOutputModel(BaseModel):
    requirements: List[str] = Field(
        description="List of achievement-framed requirements extracted from job summary"
    )

class QualityCheckerAgentOutputModel(BaseModel):
    kind: Literal["QUALITY_CHECK"] = "QUALITY_CHECK"

    status: Literal["PASS", "RETRY"] = Field(
        ...,
        description="Either 'PASS' or 'RETRY'"
    )

    critique: Optional[str] = Field(
        default=None,
        description="If RETRY, actionable feedback for revision; otherwise 'N/A'"
    )

    specific_fix_instructions: Optional[str] = Field(
        default=None,
        description="If RETRY, one or more concrete example fixes; otherwise 'N/A'"
    )
    
class CVTaskAgentOutputModel(BaseModel):
    blueprint: str = Field(
        description="The blueprint created from the summarised job description "
    )

class CVAgentOutputModel(BaseModel):
    document: str = Field(description="Final, human-sounding cover letter text suitable for a high-level candidate")

class CLTaskAgentOutputModel(BaseModel):
    blueprint: str = Field(
        description="The blueprint created from the summarised job description "
    )

class CLAgentOutputModel(BaseModel):
    document: str = Field(description="Final, human-sounding cover letter text suitable for a high-level candidate")
