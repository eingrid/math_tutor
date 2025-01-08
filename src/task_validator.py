from typing import Any, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
import wolframalpha
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class TaskValidatorWithRAG:
    def __init__(self, llm: Any, WOLFRAM_KEY):
        # Initialize Wolfram Alpha directly
        self.client = wolframalpha.Client(WOLFRAM_KEY)

        embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3",model_kwargs = {"trust_remote_code": True})

        self.vector_store = Chroma(
            collection_name="test",
            persist_directory="./chroma_langchain_db",
            embedding_function=embeddings,
        )
        # Define the tool
        @tool
        def wolfram_tool(wolfram_input: str) -> str:
            """Use Wolfram Alpha for complex mathematical computations. Best suited for:
            - Matrix operations (multiplication, determinants, eigenvalues)
            - Calculus (derivatives, integrals, limits)
            - Linear algebra (systems of equations, vector operations)
            - Statistical calculations
            - Numerical computations requiring high precision
            
            Example inputs:
            - "Calculate matrix multiplication of [[1,2],[3,4]] and [[5,6],[7,8]]"
            - "Find derivative of x^3 * sin(x)"
            - "Solve system: 3x + 2y = 12, 5x - y = 7"
            """
            try:
                res = self.client.query(wolfram_input)
                results = 'Wolfram results: \n'
                
                for pod in res.pods:
                    if pod.title == 'Result' or pod.title == 'Solution':
                        for subpod in pod.subpods:
                            results += subpod.plaintext + '\n'
                
                # If no Result/Solution pods found, get all numerical results
                if results == 'Wolfram results: \n':
                    for txt_res in res.results:
                        if isinstance(txt_res['subpod'], list):
                            for subpod in txt_res['subpod']:
                                results += subpod['plaintext'] + '\n'
                        elif isinstance(txt_res['subpod'], dict):
                            results += txt_res['subpod']['plaintext'] + '\n'
                
                return results.strip()
            except Exception as e:
                print("Error in alpha, query", wolfram_input)
                return f"Error in Wolfram computation: {str(e)}"
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise mathematical problem checker. Your task is to solve the given mathematical problem with detailed steps and compare with student's answer.


1. Analyze the given problem carefully
2. Plan the solution approach
3. Use the wolfram_tool for calculations, showing all steps
4. Format the final answer in a standardized way:
   - Use decimal notation for numbers
   - Include units if specified in the problem
   - Express complex numbers as a+bi
   - Show vectors in [x,y,z] format
5. Compare the calculated answer with the student's answer, if student's answer is correct then Coincides: True
6. Return a response in this format:

Solution:
[Detailed solution steps]

Correct Answer: [Your calculated answer]
Student Answer: [Given answer]
Coincides: True/False

Remember:
- Be precise with numerical comparisons
- Consider acceptable rounding differences
- Handle different but equivalent forms of answers
- Show clear reasoning for marking answers incorrect
"""),
            ("human", "Here is the problem {input}. \n And here is student's answer {answer}. Here are similar problems that might help with solutions {similar_problems}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Set up tools
        tools = [wolfram_tool]
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def validate_problem(self, problem: str, answer : str, k = 5) -> str:
        """
        Solve a mathematical problem using Wolfram Alpha.
        
        Args:
            problem (str): The mathematical problem to solve
            answer (str): The student's answer to the problem
            k (int): Number of problems to use for RAG
            
        Returns:
            str: The solution from the agent
        """
        try:
            similar_problems = self.retrieve_and_format(problem)
            result = self.agent_executor.invoke({"input": problem, "answer":answer, "similar_problems": similar_problems})
            return result["output"]
        except Exception as e:
            return f"Error solving problem: {str(e)}"
        
    def retrieve_and_format(self, problem: str) -> str:
        retrieved_docs = self.vector_store.similarity_search(problem)
        result = ""
        for i,doc in enumerate(retrieved_docs):
            result += f"Similar Problem {i+1} : {doc.page_content}"
            result += f"\n Correct Solution : {doc.metadata['solution']}"
            result += f"\n Answer: {doc.metadata['answer']}\n\n"
        return result





class TaskValidator:
    def __init__(self, llm: Any):
        # Initialize Wolfram Alpha
        # Initialize Wolfram Alpha directly
        self.client = wolframalpha.Client("63TH74-69PKKUTVGK")
        
        # Define the tool
        @tool
        def wolfram_tool(wolfram_input: str) -> str:
            """Use Wolfram Alpha for complex mathematical computations. Best suited for:
            - Matrix operations (multiplication, determinants, eigenvalues)
            - Calculus (derivatives, integrals, limits)
            - Linear algebra (systems of equations, vector operations)
            - Statistical calculations
            - Numerical computations requiring high precision
            
            Example inputs:
            - "Calculate matrix multiplication of [[1,2],[3,4]] and [[5,6],[7,8]]"
            - "Find derivative of x^3 * sin(x)"
            - "Solve system: 3x + 2y = 12, 5x - y = 7"
            """
            try:
                res = self.client.query(wolfram_input)
                results = 'Wolfram results: \n'
                
                for pod in res.pods:
                    if pod.title == 'Result' or pod.title == 'Solution':
                        for subpod in pod.subpods:
                            results += subpod.plaintext + '\n'
                
                # If no Result/Solution pods found, get all numerical results
                if results == 'Wolfram results: \n':
                    for txt_res in res.results:
                        if isinstance(txt_res['subpod'], list):
                            for subpod in txt_res['subpod']:
                                results += subpod['plaintext'] + '\n'
                        elif isinstance(txt_res['subpod'], dict):
                            results += txt_res['subpod']['plaintext'] + '\n'
                
                return results.strip()
            except Exception as e:
                print("Error in alpha, query", wolfram_input)
                return f"Error in Wolfram computation: {str(e)}"
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise mathematical problem checker. Your task is to solve the given mathematical problem with detailed steps and compare with student's answer.


1. Analyze the given problem carefully
2. Plan the solution approach
3. Use the wolfram_tool for calculations, showing all steps
4. Format the final answer in a standardized way:
   - Use decimal notation for numbers
   - Include units if specified in the problem
   - Express complex numbers as a+bi
   - Show vectors in [x,y,z] format
5. Compare the calculated answer with the student's answer, if student's answer is correct then Coincides: True
6. Return a response in this format:

Solution:
[Detailed solution steps]

Correct Answer: [Your calculated answer]
Student Answer: [Given answer]
Coincides: True/False

Remember:
- Be precise with numerical comparisons
- Consider acceptable rounding differences
- Handle different but equivalent forms of answers
- Show clear reasoning for marking answers incorrect
"""),
            ("human", "Here is the problem {input}. \n And here is student's answer {answer}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Set up tools
        tools = [wolfram_tool]
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def validate_problem(self, problem: str, answer : str) -> str:
        """
        Solve a mathematical problem using Wolfram Alpha.
        
        Args:
            problem (str): The mathematical problem to solve
            
        Returns:
            str: The solution from the agent
        """
        try:
            result = self.agent_executor.invoke({"input": problem, "answer":answer})
            return result["output"]
        except Exception as e:
            return f"Error solving problem: {str(e)}"