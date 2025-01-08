from typing import Any, Dict
from dataclasses import dataclass
# from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Literal

class MathTask(BaseModel):
    problem_field: list[str] = Field(
        description=(
            "List multiple types of problems that can be created within the given topic. "
            "These problems should vary in approach and application, exploring different aspects of the topic. "
            "After listing, explicitly choose one problem type that aligns with the requested difficulty level and justify the choice. "
        )
    )
    problem_idea: str = Field(description = "Think about problem you will generate, explain why this problem for this difficulty")
    problem: str = Field(description="Problem statement")
    solution: str = Field(description="Step by step solution of the problem")
    answer: str = Field(
        description="Answer to the problem - must be a number, vector, or array. "
    )
    possible_answers: list[str] = Field(
        description="Incorrect answers for quiz - each must be a number, vector, or array. "
    )

class MathQuestionGenerator:
    def __init__(self, llm: Any):

        with open("prompts/math_generator_prompt.txt") as f:
            template = f.read()

        parser = JsonOutputParser(pydantic_object=MathTask)

        prompt = PromptTemplate(
            template = template,
            input_variables=["topic","problem_description"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        self.chain = prompt | llm | parser

    def generate_problem(self, topic: str, problem_type: Literal["task", "quiz"] = "task") -> str:
        problem_description = ""
        if problem_type == "quiz":
            problem_description = """Generate quiz with four options, where only one answer is correct."""
        elif problem_type == "task":
            problem_description = "Generate task with and only one correct answer."

        print("Start generating...")
        return self.chain.invoke({"topic":topic, "problem_description":problem_description})
    

class MathQuestionGeneratorWithRAG:
    def __init__(self, llm: Any):

        with open("prompts/math_generator_prompt.txt") as f:
            template = f.read()

        parser = JsonOutputParser(pydantic_object=MathTask)

        prompt = PromptTemplate(
            template = template,
            input_variables=["topic","problem_description", "similar_tasks"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        self.chain = prompt | llm | parser

    def generate_problem(self, topic: str, problem_type: Literal["task", "quiz"] = "task") -> str:
        problem_description = ""
        if problem_type == "quiz":
            problem_description = """Generate quiz with four options, where only one answer is correct."""
        elif problem_type == "task":
            problem_description = "Generate task with correct and only one answer."

        print("start generating")
        return self.chain.invoke({"topic":topic, "problem_description":problem_description})
    
