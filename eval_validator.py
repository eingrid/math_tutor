from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from src.task_validator import TaskValidator, TaskValidatorWithRAG
from dotenv import load_dotenv
import os


def validate_translated_problems(validator, test_path: str = "MATH_TRANSLATED/test",
                               wrong_test_path: str = "MATH_TRANSLATED/test_wrong_answers") -> Dict:
    """
    Validate all translated problems in both test and wrong_test directories using the TaskValidator.
    
    Args:
        validator: TaskValidator instance
        test_path: Path to the test directory containing correct translated problems
        wrong_test_path: Path to the directory containing problems with wrong answers
                        (we expect LLM to return "Coincides: False" for these)
    
    Returns:
        Dict containing validation statistics and results for both correct and incorrect examples
    """
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_path": test_path,
            "wrong_test_path": wrong_test_path
        },
        "correct_examples": {
            "total": 0,
            "correct": 0,  # When LLM says "Coincides: True"
            "incorrect": 0,
            "errors": 0,
            "by_subject": {},
            "detailed_results": []
        },
        "wrong_examples": {
            "total": 0,
            "correct": 0,  # When LLM correctly says "Coincides: False"
            "incorrect": 0, # When LLM incorrectly says "Coincides: True"
            "errors": 0,
            "by_subject": {},
            "detailed_results": []
        }
    }
    
    def process_directory(dir_path: str, result_key: str) -> None:
        """Helper function to process files in a directory and update corresponding metrics"""
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            return

        for subject_dir in os.listdir(dir_path):
            subject_path = os.path.join(dir_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            # Initialize subject statistics if not exists
            if subject_dir not in results[result_key]["by_subject"]:
                results[result_key]["by_subject"][subject_dir] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "errors": 0
                }
            
            # Process all JSON files in subject directory
            json_files = list(Path(subject_path).glob("*.json"))
            for json_file in tqdm(json_files, desc=f"Validating {result_key} - {subject_dir}"):
                try:
                    # Load translated problem
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Validate the problem
                    validation_result = validator.validate_problem(
                        problem=data['task'],
                        answer=data['answer']
                    )
                    
                    # Parse validation result
                    llm_says_correct = "Coincides: True" in validation_result
                    
                    # For wrong examples, we expect "Coincides: False"
                    # For correct examples, we expect "Coincides: True"
                    is_correct = (
                        llm_says_correct if result_key == "correct_examples"
                        else not llm_says_correct  # For wrong examples, "False" is the correct response
                    )
                    
                    # Update statistics
                    results[result_key]["total"] += 1
                    results[result_key]["by_subject"][subject_dir]["total"] += 1
                    
                    if is_correct:
                        results[result_key]["correct"] += 1
                        results[result_key]["by_subject"][subject_dir]["correct"] += 1
                    else:
                        results[result_key]["incorrect"] += 1
                        results[result_key]["by_subject"][subject_dir]["incorrect"] += 1
                    
                    # Store detailed result
                    results[result_key]["detailed_results"].append({
                        "file": str(json_file),
                        "subject": subject_dir,
                        "problem": data['task'],
                        "solution": data.get('solution', ''),
                        "answer": data['answer'],
                        "llm_says_correct": llm_says_correct,
                        "validation_correct": is_correct,  # Whether LLM gave expected response
                        "validator_output": validation_result,
                        "expected_wrong": result_key == "wrong_examples"
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing {json_file}: {str(e)}"
                    print(error_msg)
                    results[result_key]["errors"] += 1
                    results[result_key]["by_subject"][subject_dir]["errors"] += 1
                    results[result_key]["detailed_results"].append({
                        "file": str(json_file),
                        "subject": subject_dir,
                        "error": error_msg,
                        "expected_wrong": result_key == "wrong_examples"
                    })
    
    # Process both directories
    process_directory(test_path, "correct_examples")
    if os.path.exists(wrong_test_path):
        process_directory(wrong_test_path, "wrong_examples")
    
    # Add summary metrics
    results["summary"] = {
        "correct_examples_accuracy": (results["correct_examples"]["correct"] / 
                                    results["correct_examples"]["total"] if results["correct_examples"]["total"] > 0 else 0),
        "wrong_examples_detection": (results["wrong_examples"]["correct"] / 
                                   results["wrong_examples"]["total"] if results["wrong_examples"]["total"] > 0 else 0),
        "total_processed": results["correct_examples"]["total"] + results["wrong_examples"]["total"],
        "total_errors": results["correct_examples"]["errors"] + results["wrong_examples"]["errors"]
    }
    
    return results

def print_validation_report(results: Dict, include_detailed: bool = False) -> None:
    """
    Print a formatted report of the validation results.
    
    Args:
        results: Dictionary containing validation results
        include_detailed: Whether to include detailed results in the report
    """
    print("\n=== Validation Report ===")
    print(f"\nTimestamp: {results['metadata']['timestamp']}")
    
    # Correct examples stats
    if results["correct_examples"]["total"] > 0:
        print("\nCorrect Examples Dataset:")
        print(f"Total problems: {results['correct_examples']['total']}")
        print(f"Correctly validated (LLM said True): {results['correct_examples']['correct']} "
              f"({results['correct_examples']['correct']/results['correct_examples']['total']*100:.1f}%)")
        print(f"Incorrectly validated (LLM said False): {results['correct_examples']['incorrect']} "
              f"({results['correct_examples']['incorrect']/results['correct_examples']['total']*100:.1f}%)")
        print(f"Errors: {results['correct_examples']['errors']}")
    
    # Wrong examples stats
    if results["wrong_examples"]["total"] > 0:
        print("\nWrong Examples Dataset:")
        print(f"Total problems: {results['wrong_examples']['total']}")
        print(f"Correctly identified as wrong (LLM said False): {results['wrong_examples']['correct']} "
              f"({results['wrong_examples']['correct']/results['wrong_examples']['total']*100:.1f}%)")
        print(f"Incorrectly validated (LLM said True): {results['wrong_examples']['incorrect']} "
              f"({results['wrong_examples']['incorrect']/results['wrong_examples']['total']*100:.1f}%)")
        print(f"Errors: {results['wrong_examples']['errors']}")

    # Results by subject for each dataset
    for dataset in ["correct_examples", "wrong_examples"]:
        if results[dataset]["total"] > 0:
            print(f"\n{dataset.replace('_', ' ').title()} by Subject:")
            for subject, stats in results[dataset]["by_subject"].items():
                if stats["total"] > 0:
                    correct_rate = stats["correct"]/stats["total"]*100
                    print(f"\n{subject}:")
                    print(f"  Total: {stats['total']}")
                    if dataset == "correct_examples":
                        print(f"  Correct validations (True): {stats['correct']} ({correct_rate:.1f}%)")
                    else:
                        print(f"  Correct identifications (False): {stats['correct']} ({correct_rate:.1f}%)")
                    print(f"  Incorrect validations: {stats['incorrect']}")
                    print(f"  Errors: {stats['errors']}")

    print("\nSummary Metrics:")
    print(f"Total problems processed: {results['summary']['total_processed']}")
    print(f"Correct examples accuracy: {results['summary']['correct_examples_accuracy']*100:.1f}%")
    if results['wrong_examples']['total'] > 0:
        print(f"Wrong examples detection rate: {results['summary']['wrong_examples_detection']*100:.1f}%")
    print(f"Total errors: {results['summary']['total_errors']}")

def save_detailed_results(results: Dict, output_file: str = "validation_results.json") -> None:
    """
    Save detailed validation results to a JSON file.
    
    Args:
        results: Dictionary containing validation results
        output_file: Path to the output JSON file
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":


    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")


    json_res = "new_eval.json"
    llm = ChatOpenAI(
        api_key=OPENAI_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    validator = TaskValidatorWithRAG(llm)

    # Run validation
    results = validate_translated_problems(validator)

    # Print report
    print_validation_report(results)

    # Save detailed results
    save_detailed_results(results,json_res)

