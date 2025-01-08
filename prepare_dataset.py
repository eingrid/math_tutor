import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import glob
from pathlib import Path
import random
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os



class TranslatedMathTask(BaseModel):
    task: str = Field(description="Ukrainian translation of the math problem")
    solution: str = Field(description="Ukrainian translation of the solution")
    answer: str = Field(description="Numerical answer extracted from the solution - must contain only the final numerical value")
    has_numerical_answer: bool = Field(description="Boolean indicating whether the problem has a numerical answer")

class MathTranslator:
    def __init__(self, llm: Any):
        template = """You are a professional mathematics translator. Translate the following math problem and solution to Ukrainian.
        Keep all mathematical notations (LaTeX) unchanged. Extract the numerical answer from the solution.
        Set has_numerical_answer to true only if the final answer is a number (can include decimals, fractions in decimal form, or negative numbers).
        Set it to false for symbolic answers, expressions, or non-numeric solutions.
        
        Original problem: {problem}
        Original solution: {solution}
        
        {format_instructions}
        """
        
        parser = JsonOutputParser(pydantic_object=TranslatedMathTask)
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem", "solution"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.chain = prompt | llm | parser

    def translate_problem(self, problem: str, solution: str) -> Dict:
        return self.chain.invoke({"problem": problem, "solution": solution})

def get_dataset_stats(base_path: str, train_size: int, test_size: int) -> Dict[str, Dict[str, int]]:
    stats = {"train": {}, "test": {}}
    
    # Count available files
    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            for subject in os.listdir(split_path):
                subject_path = os.path.join(split_path, subject)
                if os.path.isdir(subject_path):
                    num_files = len(glob.glob(os.path.join(subject_path, "*.json")))
                    stats[split][subject] = min(num_files, 
                                             train_size if split == "train" else test_size)
    
    return stats

def process_dataset(
    llm: Any,
    base_path: str = "MATH",
    output_path: str = "MATH_TRANSLATED",
    train_size: int = 100,
    test_size: int = 50
) -> None:
    # Create translator
    translator = MathTranslator(llm)
    
    # Get dataset statistics
    stats = get_dataset_stats(base_path, train_size, test_size)
    
    # Print statistics
    print("\nDataset Statistics:")
    for split, subjects in stats.items():
        print(f"\n{split.upper()}:")
        for subject, count in subjects.items():
            print(f"{subject}: {count} samples")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with the translation? (yes/no): ")
    if response.lower() != "yes":
        print("Operation cancelled.")
        return
    
    # Process the dataset
    for split in ["train", "test"]:
        for subject in stats[split].keys():
            processed_count = 0
            saved_count = 0
            target_count = stats[split][subject]
            
            input_dir = os.path.join(base_path, split, subject)
            output_dir = os.path.join(output_path, split, subject)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all json files
            json_files = glob.glob(os.path.join(input_dir, "*.json"))
            random.shuffle(json_files)
            
            for file_path in json_files:
                if saved_count >= target_count:  # Changed from processed_count to saved_count
                    break
                    
                try:
                    # Read original problem
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Translate problem
                    translated = translator.translate_problem(
                        data['problem'].replace("\\","\\\\"),
                        data['solution'].replace("\\","\\\\")
                    )
                    
                    processed_count += 1
                    
                    # Only save and count problems with numerical answers
                    if translated['has_numerical_answer']:
                        # Save translated problem
                        output_file = os.path.join(output_dir, os.path.basename(file_path))
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(translated, f, ensure_ascii=False, indent=2)
                        
                        saved_count += 1
                        print(f"Saved ({saved_count}/{target_count}): {file_path}")
                    else:
                        print(f"Skipped (no numerical answer): {file_path}")
                    
                    print(f"Processed total: {processed_count}, Saved with numerical answers: {saved_count}")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    # Import your LLM here
    llm = ChatOpenAI(api_key=OPENAI_KEY,
                 model="gpt-4o-mini",
                 temperature=0)
    
    # Set parameters
    TRAIN_SIZE = 10  # Number of samples per subject in train
    TEST_SIZE = 5    # Number of samples per subject in test
    
    # Process dataset
    process_dataset(
        llm=llm,
        base_path="MATH",
        output_path="MATH_TRANSLATED",
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE
    )