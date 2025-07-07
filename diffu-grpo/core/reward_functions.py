import re
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardOutput:
    """Structured output for reward functions."""
    rewards: List[float]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, name: str, weight: float = 1.0, 
                 enable_logging: bool = False, log_probability: float = 0.1):
        self.name = name
        self.weight = weight
        self.enable_logging = enable_logging
        self.log_probability = log_probability
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
    
    @abstractmethod
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute reward for given prompts and completions."""
        pass
    
    def __call__(self, prompts: List[Any], completions: List[Any], 
                 **kwargs) -> List[float]:
        """Call interface for compatibility with existing code."""
        self._call_count += 1
        
        try:
            result = self.compute_reward(prompts, completions, **kwargs)
            
            if result.success:
                self._success_count += 1
                # Log reward details occasionally to monitor performance
                if self.enable_logging and (self._success_count % max(1, int(1/self.log_probability))) == 0:
                    self._log_reward_details(prompts, completions, result, **kwargs)
                return result.rewards
            else:
                self._error_count += 1
                logger.warning(f"Reward function {self.name} failed: {result.error_message}")
                return [0.0] * len(completions)
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Reward function {self.name} crashed: {str(e)}")
            # Return zero rewards for all completions on error
            return [0.0] * len(completions)
    
    def _log_reward_details(self, prompts: List[Any], completions: List[Any], 
                           result: RewardOutput, **kwargs):
        """Log detailed reward computation information."""
        # Focus on reward statistics rather than individual completions
        avg_reward = np.mean(result.rewards) if result.rewards else 0.0
        max_reward = max(result.rewards) if result.rewards else 0.0
        min_reward = min(result.rewards) if result.rewards else 0.0
        
        logger.info(f"[{self.name}] Reward Stats - Avg: {avg_reward:.3f}, Max: {max_reward:.3f}, Min: {min_reward:.3f}")
        if result.metadata:
            # Only log key metadata, not everything
            key_metrics = {}
            for key, value in result.metadata.items():
                if key in ['accuracy', 'correct_count', 'match_count', 'avg_score']:
                    key_metrics[key] = value
            if key_metrics:
                logger.info(f"[{self.name}] Metrics: {key_metrics}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward function statistics."""
        return {
            'name': self.name,
            'weight': self.weight,
            'call_count': self._call_count,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'success_rate': self._success_count / self._call_count if self._call_count > 0 else 0.0
        }


class FormatRewardFunction(RewardFunction):
    """Base class for format-based reward functions."""
    
    def __init__(self, name: str, pattern: str, reward_value: float = 0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.pattern = pattern
        self.reward_value = reward_value
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute format-based rewards - faithful to original."""
        try:
            responses = [completion[0]["content"] for completion in completions]
            matches = [re.match(self.pattern, r) for r in responses]
            rewards = [0.5 if match else 0.0 for match in matches]
            
            return RewardOutput(
                rewards=rewards,
                metadata={
                    'pattern': self.pattern,
                    'match_count': sum(1 for m in matches if m),
                    'total_count': len(responses)
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _extract_responses(self, completions: List[Any]) -> List[str]:
        """Extract response strings from completions."""
        responses = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    responses.append(completion[0]["content"])
                else:
                    responses.append(str(completion[0]))
            else:
                responses.append(str(completion))
        return responses


class XMLFormatRewardFunction(FormatRewardFunction):
    """Reward function for XML format validation."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="xml_format",
            pattern=r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
            **kwargs
        )


class StrictXMLFormatRewardFunction(FormatRewardFunction):
    """Reward function for strict XML format validation."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="strict_xml_format",
            pattern=r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$",
            **kwargs
        )


class XMLCountRewardFunction(RewardFunction):
    """Reward function that counts XML tag occurrences."""
    
    def __init__(self, **kwargs):
        super().__init__(name="xml_count", **kwargs)
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute XML count-based rewards - faithful to original."""
        try:
            contents = [completion[0]["content"] for completion in completions]
            rewards = [self._count_xml(c) for c in contents]
            
            return RewardOutput(
                rewards=rewards,
                metadata={'avg_count': np.mean(rewards) if rewards else 0.0},
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _count_xml(self, text: str) -> float:
        """Count XML tags - faithful to original count_xml function."""
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count
    
    def _extract_responses(self, completions: List[Any]) -> List[str]:
        """Extract response strings from completions."""
        responses = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    responses.append(completion[0]["content"])
                else:
                    responses.append(str(completion[0]))
            else:
                responses.append(str(completion))
        return responses


class CorrectnessRewardFunction(RewardFunction):
    """Reward function for correctness validation."""
    
    def __init__(self, correct_reward: float = 2.0, **kwargs):
        super().__init__(name="correctness", **kwargs)
        self.correct_reward = correct_reward
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      answer: List[str], **kwargs) -> RewardOutput:
        """Compute correctness-based rewards - faithful to original."""
        try:
            responses = [completion[0]["content"] for completion in completions]
            q = prompts[0][-1]["content"]
            extracted_responses = [self._extract_xml_answer(r) for r in responses]

            # Sample one completion for detailed logging (reduce noise)
            if self.enable_logging and np.random.random() < 0.1:  # 10% chance
                logger.debug(f"[{self.name}] Sample - Ground truth: {answer[0][:50]}...")
                logger.debug(f"[{self.name}] Sample - Extracted: {extracted_responses[0][:50]}...")
                logger.debug(f"[{self.name}] Sample - Correct: {extracted_responses[0] == answer[0]}")
            
            rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
            correct_count = sum(1 for r in rewards if r > 0.0)
            
            return RewardOutput(
                rewards=rewards,
                metadata={
                    'correct_count': correct_count,
                    'total_count': len(responses),
                    'accuracy': correct_count / len(responses) if responses else 0.0
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _extract_responses(self, completions: List[Any]) -> List[str]:
        """Extract response strings from completions."""
        responses = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    responses.append(completion[0]["content"])
                else:
                    responses.append(str(completion[0]))
            else:
                responses.append(str(completion))
        return responses
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags, handling both plain text and boxed format."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            answer = answer.strip()
            
            # If the answer contains \boxed{}, extract the content inside
            if "\\boxed" in answer:
                try:
                    # Import here to avoid circular imports
                    from math500_utils import remove_boxed, last_boxed_only_string
                    answer = remove_boxed(last_boxed_only_string(answer))
                except:
                    # If boxed extraction fails, try simple regex
                    import re
                    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
                    if boxed_match:
                        answer = boxed_match.group(1)
            
            return answer.strip()
        except:
            return ""
    
    def _is_valid_answer_format(self, extracted_answer: str) -> bool:
        """Check if the extracted answer has a valid format."""
        if not extracted_answer:
            return False
        
        # Basic validation: non-empty, reasonable length
        if len(extracted_answer.strip()) == 0:
            return False
        
        # For sudoku-like tasks, check if it contains reasonable content
        if len(extracted_answer) > 1000:  # Too long
            return False
        
        # Contains some alphanumeric content
        if not any(c.isalnum() for c in extracted_answer):
            return False
        
        return True


class IntegerRewardFunction(RewardFunction):
    """Reward function for integer format validation."""
    
    def __init__(self, reward_value: float = 0.5, **kwargs):
        super().__init__(name="integer_format", **kwargs)
        self.reward_value = reward_value
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute integer format rewards - faithful to original."""
        try:
            responses = [completion[0]["content"] for completion in completions]
            extracted_responses = [self._extract_xml_answer(r) for r in responses]
            rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
            
            return RewardOutput(
                rewards=rewards,
                metadata={
                    'integer_count': sum(1 for r in rewards if r > 0.0),
                    'total_count': len(responses)
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _extract_responses(self, completions: List[Any]) -> List[str]:
        """Extract response strings from completions."""
        responses = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    responses.append(completion[0]["content"])
                else:
                    responses.append(str(completion[0]))
            else:
                responses.append(str(completion))
        return responses
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags, handling both plain text and boxed format."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            answer = answer.strip()
            
            # If the answer contains \boxed{}, extract the content inside
            if "\\boxed" in answer:
                try:
                    # Import here to avoid circular imports
                    from math500_utils import remove_boxed, last_boxed_only_string
                    answer = remove_boxed(last_boxed_only_string(answer))
                except:
                    # If boxed extraction fails, try simple regex
                    import re
                    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
                    if boxed_match:
                        answer = boxed_match.group(1)
            
            return answer.strip()
        except:
            return ""


class CountdownRewardFunction(RewardFunction):
    """Reward function for countdown task validation."""
    
    def __init__(self, correct_score: float = 1.0, format_score: float = 0.1, **kwargs):
        super().__init__(name="countdown", **kwargs)
        self.correct_score = correct_score
        self.format_score = format_score
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      target: List[int], numbers: List[List[int]], **kwargs) -> RewardOutput:
        """Compute countdown task rewards."""
        try:
            responses = self._extract_responses(completions)
            rewards = []
            correct_count = 0
            
            for i, response in enumerate(responses):
                ground_truth = {"target": target[i], "numbers": numbers[i]}
                score = self._compute_score(response, ground_truth)
                rewards.append(score)
                if score >= self.correct_score:
                    correct_count += 1
            
            return RewardOutput(
                rewards=rewards,
                metadata={
                    'correct_count': correct_count,
                    'total_count': len(responses),
                    'accuracy': correct_count / len(responses) if responses else 0.0
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _extract_responses(self, completions: List[Any]) -> List[str]:
        """Extract response strings from completions."""
        responses = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    responses.append(completion[0]["content"])
                else:
                    responses.append(str(completion[0]))
            else:
                responses.append(str(completion))
        return responses
    
    def _compute_score(self, solution_str: str, ground_truth: Dict[str, Any]) -> float:
        """Compute score for countdown task."""
        target = ground_truth["target"]
        numbers = ground_truth["numbers"]
        
        # Extract equation from solution
        equation = self._extract_solution(solution_str)
        
        if equation is None:
            return 0.0
        
        # Validate equation uses correct numbers
        if not self._validate_equation(equation, numbers):
            return self.format_score
        
        # Evaluate equation
        try:
            result = self._evaluate_equation(equation)
            if result is None:
                return self.format_score
            
            if abs(result - target) < 1e-5:
                return self.correct_score
            else:
                return self.format_score
        except:
            return self.format_score
    
    def _extract_solution(self, solution_str: str) -> Optional[str]:
        """Extract solution from answer tags."""
        answer_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(answer_pattern, solution_str, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    def _validate_equation(self, equation_str: str, available_numbers: List[int]) -> bool:
        """Validate equation uses correct numbers."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            return sorted(numbers_in_eq) == sorted(available_numbers)
        except:
            return False
    
    def _evaluate_equation(self, equation_str: str) -> Optional[float]:
        """Safely evaluate equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                return None
            return eval(equation_str, {"__builtins__": None}, {})
        except:
            return None


class SudokuRewardFunction(RewardFunction):
    """Reward function for sudoku task validation - faithful to original implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(name="sudoku", **kwargs)
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute sudoku rewards faithful to original sudoku_reward_func."""
        try:
            # Extract responses same way as original
            if (isinstance(completions[0], list) and 
                isinstance(completions[0][0], dict) and 
                "content" in completions[0][0]):
                responses = [completion[0]["content"] for completion in completions]
            else:
                responses = completions
            
            scores = []
            for i, response in enumerate(responses):
                puzzle = kwargs["puzzle"][i]
                ground_truth = kwargs["solution"][i]
                solution = self._extract_answer_sudoku(response)
                
                score = 0.0 if solution is None else self._validate_sudoku_solution(solution, ground_truth, puzzle)
                scores.append(score)
                
                # Sample logging for debugging (reduce noise)
                if self.enable_logging and np.random.random() < 0.05:  # 5% chance
                    logger.debug(f"[{self.name}] Sample - Puzzle length: {len(puzzle)}")
                    logger.debug(f"[{self.name}] Sample - Solution length: {len(solution) if solution else 0}")
                    logger.debug(f"[{self.name}] Sample - Score: {score:.4f}")
            
            return RewardOutput(
                rewards=scores,
                metadata={
                    'avg_score': np.mean(scores) if scores else 0.0,
                    'total_count': len(responses)
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _extract_answer_sudoku(self, solution_str: str) -> Optional[str]:
        """Extract answer from sudoku solution - faithful to original."""
        answer_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(answer_pattern, solution_str, re.DOTALL)
        if matches:
            return "".join(char for char in matches[-1].strip() if char.isdigit())
        return None
    
    def _validate_sudoku_solution(self, solution_str: str, ground_truth: str, puzzle: str) -> float:
        """Validate sudoku solution - faithful to original."""
        if solution_str is None or len(solution_str) == 0:
            return 0.0

        if len(solution_str) < 16:
            # Pad with zeros if too short
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            # Truncate if too long
            solution_str = solution_str[:16]

        empty_indices = [i for i in range(16) if puzzle[i] == "0"]

        if empty_indices:
            correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
            return correct_cells / len(empty_indices)
        return 0.0


class BoxedFormatRewardFunction(RewardFunction):
    """Reward function specifically for boxed format within answer tags."""
    
    def __init__(self, reward_value: float = 0.5, **kwargs):
        super().__init__(name="boxed_format", **kwargs)
        self.reward_value = reward_value
    
    def compute_reward(self, prompts: List[Any], completions: List[Any], 
                      **kwargs) -> RewardOutput:
        """Compute boxed format rewards."""
        try:
            responses = [completion[0]["content"] for completion in completions]
            rewards = []
            
            for r in responses:
                reward = 0.0
                try:
                    # Extract content between answer tags
                    answer_content = r.split("<answer>")[1].split("</answer>")[0]
                    
                    # Check if it contains \boxed format
                    if "\\boxed{" in answer_content and "}" in answer_content:
                        reward += self.reward_value
                    elif "\\boxed" in answer_content:
                        reward += self.reward_value * 0.5  # Partial credit for having \boxed but maybe malformed
                    
                except (IndexError, ValueError):
                    pass
                
                rewards.append(reward)
            
            return RewardOutput(
                rewards=rewards,
                metadata={
                    'boxed_count': sum(1 for r in rewards if r > 0.0),
                    'total_count': len(responses)
                },
                success=True
            )
            
        except Exception as e:
            return RewardOutput(
                rewards=[0.0] * len(completions),
                metadata={},
                success=False,
                error_message=str(e)
            )


class RewardFunctionManager:
    """Manager for multiple reward functions."""
    
    def __init__(self, reward_functions: List[RewardFunction], weights: Optional[List[float]] = None):
        self.reward_functions = reward_functions
        self.weights = weights or [1.0] * len(reward_functions)
        
        if len(self.weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
    
    def compute_rewards(self, prompts: List[Any], completions: List[Any], 
                       **kwargs) -> Tuple[List[float], Dict[str, Any]]:
        """Compute weighted rewards from all functions."""
        all_rewards = []
        all_metadata = {}
        
        for i, reward_func in enumerate(self.reward_functions):
            try:
                rewards = reward_func(prompts, completions, **kwargs)
                weighted_rewards = [r * self.weights[i] for r in rewards]
                all_rewards.append(weighted_rewards)
                all_metadata[reward_func.name] = reward_func.get_stats()
            except Exception as e:
                logger.error(f"Error in reward function {reward_func.name}: {str(e)}")
                all_rewards.append([0.0] * len(completions))
                all_metadata[reward_func.name] = {'error': str(e)}
        
        # Combine rewards
        if all_rewards:
            combined_rewards = [sum(rewards) for rewards in zip(*all_rewards)]
        else:
            combined_rewards = [0.0] * len(completions)
        
        # Log combined reward statistics occasionally
        if hasattr(self, '_call_count'):
            self._call_count += 1
        else:
            self._call_count = 1
            
        if self._call_count % 50 == 0:  # Log every 50 calls
            avg_reward = np.mean(combined_rewards) if combined_rewards else 0.0
            max_reward = max(combined_rewards) if combined_rewards else 0.0
            logger.info(f"[RewardManager] Combined reward stats - Avg: {avg_reward:.3f}, Max: {max_reward:.3f}")
        
        return combined_rewards, all_metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all reward functions."""
        stats = {}
        for reward_func in self.reward_functions:
            stats[reward_func.name] = reward_func.get_stats()
        return stats


# Factory functions for backward compatibility
def create_reward_functions(dataset: str, enable_logging: bool = True, log_probability: float = 0.1) -> List[RewardFunction]:
    """Create reward functions for a given dataset with configurable logging."""
    if dataset == "gsm8k":
        return [
            XMLFormatRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
            StrictXMLFormatRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
            BoxedFormatRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
            CorrectnessRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
        ]
    elif dataset == "countdown":
        return [CountdownRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability)]
    elif dataset == "sudoku":
        return [
            XMLFormatRewardFunction(weight=0.5, enable_logging=enable_logging, log_probability=log_probability),
            SudokuRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
        ]
    elif dataset == "math":
        return [
            CorrectnessRewardFunction(weight=1.0, enable_logging=enable_logging, log_probability=log_probability),
            BoxedFormatRewardFunction(weight=0.5, enable_logging=enable_logging, log_probability=log_probability),
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")