from .output_models import *

PYDANTIC_REGISTRY = {
    "Summary_Agent": SummaryAgentOutputModel,
    "Router_Agent": RouterAgentOutputModel,
    "CV_Task_Agent": CVTaskAgentOutputModel,
    "CV_Agent": CVAgentOutputModel,
    "CL_Task_Agent": CLTaskAgentOutputModel,
    "CL_Agent": CLAgentOutputModel,
    "Quality_Checker_Agent": QualityCheckerAgentOutputModel,
}