from langchain_core.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from ..workflows.prompts.cover_letter_prompts import * 
from ..workflows.prompts.cv_prompts import * 
from ..workflows.prompts.system_prompts import * 

class PromptFactory:
    def __init__(self):
        pass
    
    def create_prompt(self, prompt_type: str) -> str:
        prompt = None

        if prompt_type == "Summary_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt + "\n\n" + SummaryPrompt
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Input job description: {job_description}"
                    )
                ],
                input_variables=[
                    'job_description'
                ],
            )

        elif prompt_type == "Semantic_Alignment_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt + "\n\n" + SemanticAlignmentAgentPrompt
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Input job description summary: {summary}"
                    )
                ],
                input_variables=[
                    'summary'
                ],
            )

        # COVER LETTER PATHWAY
        elif prompt_type == "CL_Task_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt + "\n\n" + CoverLetterTaskPrompt
                    ),
                    AIMessagePromptTemplate.from_template(
                        "Input from previous agent:\n{task_agent_input}"
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Retrieved Documents (Background information):\n{retrieved_documents}"
                    ),
                ],
                input_variables=[
                    'task_agent_input',
                    'retrieved_documents'
                ],
            )

        elif prompt_type == "CL_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt + "\n\n" + CoverLetterWriterPrompt
                    ),
                    AIMessagePromptTemplate.from_template(
                        "Input blueprint from task agent: {blueprint}"
                    )
                ],
                input_variables=[
                    'blueprint'
                ],
            )

        elif prompt_type == "Quality_Checker_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt + "\n\n" + CoverLetterQualityCheckerPrompt
                    ),

                    AIMessagePromptTemplate.from_template(
                        "Writing strategy blueprint:\n{blueprint}"
                    ),

                    HumanMessagePromptTemplate.from_template(
                        "Candidate cover letter to be evaluated:\n{document}"
                    ),

                    HumanMessagePromptTemplate.from_template(
                        "Retrieved Documents (Background information):\n{retrieved_documents}\n\n"
                        "Job description for this task:\n{job_description}"
                    ),
                ],
                input_variables=[
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
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt
                    ),
                    HumanMessagePromptTemplate.from_template(
                        CVWriterPrompt
                    )
                ],
                input_variables=[
                    'blueprint',
                ],
            )

        elif prompt_type == "CV_Task_Agent":
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        SystemPrompt
                    ),
                    HumanMessagePromptTemplate.from_template(
                        CVTaskPrompt
                    )
                ],
                input_variables=[
                    'summary',
                ],
            )

        return prompt