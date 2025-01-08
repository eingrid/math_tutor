from src.question_generator import MathQuestionGenerator
from src.task_validator import TaskValidator, TaskValidatorWithRAG
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
WOLFRAM_KEY = os.getenv("WOLFRAM_KEY")

llm = ChatOpenAI(
    api_key=OPENAI_KEY,
    model="gpt-4o-mini"
    )

question_generator = MathQuestionGenerator(llm)

validator = TaskValidatorWithRAG(llm,WOLFRAM_KEY)

problems_to_generate = [("лінійна алгебра, жорданова нормальна форма","task"),("лінійна алгебра, жорданова нормальна форма","quiz"),
                        ("математичний аналіз, ряди","task"), ("дискретна математика, графи", "task"), (("математичний аналіз, нескінченний ряд","task"))]


for i,topic_problem_type in enumerate(problems_to_generate):
    topic, problem_type = topic_problem_type
    is_correct = False
    while not is_correct:
        try:
            res = question_generator.generate_problem(topic,problem_type)
            validator_res = validator.validate_problem(res['problem'], res['answer'])
            is_correct = "Coincides: True" in validator_res
        except:
            is_correct = False

    print(f"Problem {i}, {topic}, {problem_type}")
    print(res['problem'])
    print("\n Answer:" + res['answer'])
    print("\n Quiz Options (wrong):", res['possible_answers'])
