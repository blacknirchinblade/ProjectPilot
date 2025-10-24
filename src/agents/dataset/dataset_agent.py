"""
Dataset Agent - Synthetic Data and ML Dataset Generation (Corrected)

This agent generates synthetic datasets for ML/DL/NLP/CV projects.
Uses temperature=0.5 for balanced creativity and precision.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import re
import json
import traceback
from typing import Dict, List, Any, Optional
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class DatasetAgent(BaseAgent):
    """
    Dataset Agent for generating synthetic data and ML datasets.
    
    Responsibilities:
    - Generate synthetic tabular data
    - Create ML datasets (train/test/val splits)
    - Generate test data for functions
    - Create image dataset specifications
    - Generate text corpora
    - Augment existing datasets
    
    Uses temperature=0.5 for creative yet structured data generation.
    """
    
    def __init__(
        self,
        name: str = "dataset_agent",
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize Dataset Agent.
        
        Args:
            name: Agent name (default: "dataset_agent")
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        role = "Expert Data Engineer for ML/DL Dataset Creation"
        super().__init__(
            name=name,
            role=role,
            agent_type="dataset",  # Uses temperature 0.5
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        logger.info(f"{self.name} ready for dataset generation tasks")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a dataset generation task.
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of dataset task
                - data: Task-specific parameters
        
        Returns:
            Dictionary with dataset generation results
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        # Ensure clients are available
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for DatasetAgent. Should be passed in __init__.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()
            
        try:
            if task_type == "generate_synthetic_data":
                return await self.generate_synthetic_data(
                    schema=data.get("schema", {}),
                    num_samples=data.get("num_samples", 100)
                )
            
            elif task_type == "create_ml_dataset":
                return await self.create_ml_dataset(
                    task_type_ml=data.get("task_type_ml", "classification"),
                    features=data.get("features", []),
                    num_samples=data.get("num_samples", 1000)
                )
            
            elif task_type == "generate_test_data":
                return await self.generate_test_data(
                    function_signature=data.get("function_signature"),
                    num_cases=data.get("num_cases", 10)
                )
            
            elif task_type == "create_image_dataset":
                return await self.create_image_dataset(
                    dataset_type=data.get("dataset_type", "classification"),
                    classes=data.get("classes", []),
                    num_samples_per_class=data.get("num_samples_per_class", 100)
                )
            
            elif task_type == "generate_text_corpus":
                return await self.generate_text_corpus(
                    domain=data.get("domain", "general"),
                    num_documents=data.get("num_documents", 50),
                    doc_length=data.get("doc_length", "medium")
                )
            
            elif task_type == "augment_dataset":
                return await self.augment_dataset(
                    original_data=data.get("original_data"),
                    augmentation_methods=data.get("augmentation_methods", [])
                )
            
            else:
                return {
                    "status": "error",
                    "task": task_type,
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Error executing dataset task '{task_type}': {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "task": task_type,
                "message": str(e)
            }
    
    async def generate_synthetic_data(
        self,
        schema: Dict[str, Any],
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate synthetic tabular data based on schema.
        """
        if not schema:
            return {
                "status": "error",
                "message": "Schema is required for synthetic data generation"
            }
        
        logger.info(f"{self.name} generating {num_samples} synthetic samples")
        
        schema_text = self._format_schema(schema)
        
        prompt_data = {
            "columns": schema_text,
            "num_rows": num_samples,
            "relationships": "None specified"
        }
        
        try:
            prompt = self.get_prompt("dataset_prompts", "generate_tabular_dummy", prompt_data)
        except ValueError:
            logger.warning("No 'generate_tabular_dummy' prompt. Using fallback.")
            prompt = f"Generate python code using faker to create a pandas dataframe with {num_samples} rows. Columns: {schema_text}."
            
        response = await self.generate_response(prompt)
        code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "generate_synthetic_data",
            "num_samples": num_samples,
            "generation_code": code,
            "columns": self._extract_columns(schema_text)
        }
    
    async def create_ml_dataset(
        self,
        task_type_ml: str = "classification",
        features: List[str] = None,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Create ML dataset with train/test/val splits.
        """
        if not features:
            features = ["feature_1 (numeric)", "feature_2 (numeric)", "target (categorical)"]
        
        logger.info(f"{self.name} creating {task_type_ml} dataset with {num_samples} samples")
        
        features_text = "\n".join([f"- {f}" for f in features])
        
        ml_dataset_prompt = f"""Create a complete ML {task_type_ml} dataset with train/test/validation splits.

Features:
{features_text}

Total Samples: {num_samples}

Generate Python code that:
- Uses sklearn.datasets or creates synthetic data using faker/numpy.
- Implements {task_type_ml} task.
- Splits data into train (70%), validation (15%), test (15%).
- Uses sklearn.model_selection.train_test_split.
- Saves each split to separate CSV files (e.g., 'train.csv', 'val.csv', 'test.csv').
- Includes proper labels/targets.

Provide complete working code with imports and file saving.
"""
        
        response = await self.generate_response(ml_dataset_prompt)
        code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "create_ml_dataset",
            "task_type_ml": task_type_ml,
            "num_samples": num_samples,
            "dataset_code": code,
            "splits": self._extract_splits(response)
        }
    
    async def generate_test_data(
        self,
        function_signature: str,
        num_cases: int = 10
    ) -> Dict[str, Any]:
        """
        Generate test data for a function.
        """
        if not function_signature:
            return {
                "status": "error",
                "message": "Function signature is required for test data generation"
            }
        
        logger.info(f"{self.name} generating {num_cases} test cases for function")
        
        test_data_prompt = f"""Generate {num_cases} comprehensive test cases for this function:

{function_signature}

For each test case, provide:
1. Test Case Number
2. Input values (with description)
3. Expected output
4. Test category (normal, edge case, error case)

Include:
- Normal/typical cases
- Edge cases (empty, null, boundary values)
- Error cases (invalid input)

Format as:
Test Case 1:
Input: function_name(arg1, arg2)
Expected: result
Category: normal

Provide clear, runnable test data.
"""
        
        response = await self.generate_response(test_data_prompt)
        test_cases = self._extract_test_cases(response)
        
        return {
            "status": "success",
            "task": "generate_test_data",
            "num_cases": len(test_cases),
            "test_cases": test_cases,
            "code": self._extract_code(response) if "```" in response else response
        }
    
    async def create_image_dataset(
        self,
        dataset_type: str = "classification",
        classes: List[str] = None,
        num_samples_per_class: int = 100
    ) -> Dict[str, Any]:
        """
        Create image dataset specification and generation code.
        """
        if not classes:
            classes = ["class_1", "class_2", "class_3"]
        
        logger.info(f"{self.name} creating {dataset_type} image dataset")
        
        classes_text = "\n".join([f"- {c}" for c in classes])
        
        prompt_data = {
            "task": dataset_type,
            "image_size": "224x224",
            "num_images": len(classes) * num_samples_per_class,
            "classes": classes_text
        }
        
        try:
            prompt = self.get_prompt("dataset_prompts", "generate_image_dummy", prompt_data)
        except ValueError:
            logger.warning("No 'generate_image_dummy' prompt. Using fallback.")
            prompt = f"Generate python code using 'Pillow' to create a dummy image dataset for {dataset_type}. Classes: {classes_text}. Num images: {prompt_data['num_images']}."

        response = await self.generate_response(prompt)
        code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "create_image_dataset",
            "dataset_type": dataset_type,
            "num_classes": len(classes),
            "total_samples": len(classes) * num_samples_per_class,
            "dataset_code": code,
            "structure": self._extract_directory_structure(response)
        }
    
    async def generate_text_corpus(
        self,
        domain: str = "general",
        num_documents: int = 50,
        doc_length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate text corpus for NLP tasks.
        """
        logger.info(f"{self.name} generating {num_documents} {doc_length} documents for {domain}")
        
        prompt_data = {
            "task": f"{domain} corpus generation",
            "characteristics": f"{doc_length} length documents about {domain}",
            "num_samples": num_documents,
            "labels": f"{domain} topics"
        }
        
        try:
            prompt = self.get_prompt("dataset_prompts", "generate_text_dummy", prompt_data)
        except ValueError:
            logger.warning("No 'generate_text_dummy' prompt. Using fallback.")
            prompt = f"Generate python code using 'faker' to create a list of {num_documents} text documents about {domain}."

        response = await self.generate_response(prompt)
        code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "generate_text_corpus",
            "domain": domain,
            "num_documents": num_documents,
            "doc_length": doc_length,
            "generation_code": code,
            "topics": self._extract_topics(response)
        }
    
    async def augment_dataset(
        self,
        original_data: str,
        augmentation_methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Augment existing dataset with various techniques.
        """
        if not original_data:
            return {
                "status": "error",
                "message": "Original data description is required for augmentation"
            }
        
        if not augmentation_methods:
            augmentation_methods = ["noise_injection", "rotation", "scaling"]
        
        logger.info(f"{self.name} augmenting dataset with {len(augmentation_methods)} methods")
        
        methods_text = "\n".join([f"- {m}" for m in augmentation_methods])
        
        augmentation_prompt = f"""Create data augmentation code for this dataset:

Original Data: {original_data}

Augmentation Methods to Apply:
{methods_text}

Generate Python code that:
- Implements each augmentation method
- Can be applied to the original data
- Increases dataset size by 2-3x
- Preserves data characteristics
- Uses appropriate libraries (torchvision, imgaug, nlpaug, etc.)
- Saves augmented data alongside original

Provide complete working code with imports and examples.
"""
        
        response = await self.generate_response(augmentation_prompt)
        code = self._extract_code(response)
        
        return {
            "status": "success",
            "task": "augment_dataset",
            "num_methods": len(augmentation_methods),
            "augmentation_code": code,
            "methods_applied": augmentation_methods,
            "augmentation_factor": self._extract_augmentation_factor(response)
        }
    
    # ==================== Helper Methods ====================
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.
        """
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return response.strip()
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """
        Format schema dictionary into readable text.
        """
        if isinstance(schema, dict) and 'columns' in schema:
            columns = schema['columns']
            if isinstance(columns, list):
                return "\n".join([f"- {col}" for col in columns])
            elif isinstance(columns, dict):
                return "\n".join([f"- {k}: {v}" for k, v in columns.items()])
        
        return str(schema)
    
    def _extract_columns(self, schema_text: str) -> List[str]:
        """
        Extract column names from schema text.
        """
        columns = []
        patterns = [r'-\s*(\w+):', r'-\s*(\w+)', r'(\w+)\s*:']
        
        for pattern in patterns:
            matches = re.findall(pattern, schema_text)
            if matches:
                columns.extend(matches)
                break
        
        return list(set(columns))[:20]
    
    def _extract_splits(self, response: str) -> Dict[str, str]:
        """
        Extract train/test/val split information.
        """
        splits = {}
        
        train_match = re.search(r'train[^:]*:\s*(\d+\.?\d*)', response, re.IGNORECASE)
        test_match = re.search(r'test[^:]*:\s*(\d+\.?\d*)', response, re.IGNORECASE)
        val_match = re.search(r'val[^:]*:\s*(\d+\.?\d*)', response, re.IGNORECASE)
        
        if train_match:
            splits['train'] = train_match.group(1)
        if test_match:
            splits['test'] = test_match.group(1)
        if val_match:
            splits['validation'] = val_match.group(1)
        
        if not splits:
            splits = {'train': '0.7', 'test': '0.2', 'validation': '0.1'}
        
        return splits
    
    def _extract_test_cases(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract test cases from response.
        """
        test_cases = []
        
        case_pattern = r'(?:Case|Test|Example)\s+\d+[:\s]*\n.*?(?:Input|input)[:\s]*([^\n]+)\n.*?(?:Expected|Output|output)[:\s]*([^\n]+)'
        matches = re.findall(case_pattern, response, re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(matches):
            test_cases.append({
                "case_id": i + 1,
                "input": match[0].strip(),
                "expected": match[1].strip()
            })
        
        if not test_cases:
            code = self._extract_code(response)
            assert_pattern = r'assert\s+.*?\((.*?)\)\s*==\s*(.*?)(?:\n|$)'
            assertions = re.findall(assert_pattern, code)
            
            for i, assertion in enumerate(assertions):
                test_cases.append({
                    "case_id": i + 1,
                    "input": assertion[0].strip(),
                    "expected": assertion[1].strip()
                })
        
        return test_cases[:20]
    
    def _extract_directory_structure(self, response: str) -> List[str]:
        """
        Extract directory structure from response.
        """
        structure = []
        dir_patterns = [r'(?:dataset|data)/[\w/]+', r'[\w]+/[\w]+', r'-\s+([\w/]+)']
        
        for pattern in dir_patterns:
            matches = re.findall(pattern, response)
            structure.extend(matches)
        
        return list(set(structure))[:30]
    
    def _extract_topics(self, response: str) -> List[str]:
        """
        Extract topics from text corpus response.
        """
        topics = []
        topic_patterns = [
            r'[Tt]opic[s]?:\s*([^\n]+)',
            r'[Cc]ategor(?:y|ies):\s*([^\n]+)',
            r'-\s*([A-Z][a-z\s]+)(?:\n|:)',
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                split_topics = re.split(r'[,;]', match)
                topics.extend([t.strip() for t in split_topics if t.strip()])
        
        return list(set(topics))[:15]
    
    def _extract_augmentation_factor(self, response: str) -> str:
        """
        Extract augmentation factor from response.
        """
        factor_patterns = [
            r'(?:augmentation\s+)?factor[:\s]*(\d+\.?\d*)',
            r'(?:increase|multiply)[^\d]*(\d+\.?\d*)[x×]?',
            r'(\d+\.?\d*)[x×]\s*(?:more|larger|increase)',
        ]
        
        for pattern in factor_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "2.0"
    
    def _extract_data_types(self, response: str) -> List[str]:
        """
        Extract data types from response.
        """
        data_types = []
        type_keywords = [
            'int', 'float', 'string', 'bool', 'datetime',
            'categorical', 'numerical', 'text', 'image'
        ]
        
        for keyword in type_keywords:
            if re.search(rf'\b{keyword}\b', response, re.IGNORECASE):
                data_types.append(keyword)
        
        return data_types