from langchain_core.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

from ..workflows.prompts.cover_letter_prompts import * 
from ..workflows.prompts.cv_prompts import * 
from ..workflows.prompts.system_prompts import * 

def _prepend_system_prompt(prompt_text):
    """Prepends the SystemPrompt to another prompt string with clear separation."""
    return f"{SystemPrompt.strip()}\n\n{prompt_text.strip()}"

class PromptFactory:
    def __init__(self):
        pass
    
    def create_prompt(self, prompt_type: str) -> str:
        prompt = None

        if prompt_type == "Summary_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(SummaryPrompt)
                    )
                ],
                input_variables = [
                    'job_description'
                ],
            )

        elif prompt_type == "Router_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(RouterPrompt)
                    )
                ],
                input_variables = [
                    'router_input'
                ],
            )

        # COVER LETTER PATHWAY
        elif prompt_type == "CL_Task_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(CoverLetterTaskPrompt)
                    )
                ],
                input_variables = [
                    'summary',
                ],
            )

        elif prompt_type == "CL_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(CoverLetterWriterPrompt)
                    )
                ],
                input_variables = [
                    'blueprint'
                ],
            )

        elif prompt_type == "Quality_Checker_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(CoverLetterQualityCheckerPrompt)
                    )
                ],
                input_variables = [
                    'retrieved_documents',
                    'job_description',
                    'blueprint',
                    'document'
                ],
            )

        # CV PATHWAY
        elif prompt_type == "CV_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(CVWriterPrompt)
                    )
                ],
                input_variables = [
                    'blueprint',
                ],
            )

        elif prompt_type == "CV_Task_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        _prepend_system_prompt(CVTaskPrompt)
                    )
                ],
                input_variables = [
                    'summary',
                ],
            )

        return prompt