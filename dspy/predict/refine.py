import copy
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, Union

import dspy
import ujson
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField, Signature


class OfferFeedback(Signature):
    """
    In the discussion, analyze why the module failed to meet the desired score threshold based on its inputs and outputs.
    Consider the trajectory, reward function, and target score. Assign blame if applicable.
    Then, prescribe concrete, actionable advice for how the module should modify its behavior on its future input
    when we retry the process, especially if it receives the same or similar inputs.
    The module will not see its own history directly, only the feedback you provide.
    Focus on specific examples from the trajectory and clear instructions for improvement.
    """

    program_code: str = InputField(desc="The code of the program (module) that we are analyzing")
    modules_defn: str = InputField(desc="The definition of the module, including its I/O signature")
    program_inputs: str = InputField(desc="The inputs to the program (module) that we are analyzing")
    program_trajectory: str = InputField(
        desc="The trajectory of the program's execution (if available, often just one step for a single module)"
    )
    program_outputs: str = InputField(desc="The outputs generated by the program (module)")
    reward_code: str = InputField(desc="The code of the reward function used for scoring")
    target_threshold: float = InputField(desc="The target threshold for the reward score")
    reward_value: float = InputField(desc="The reward score assigned to the program's output")
    discussion: str = OutputField(
        desc="Discuss why the module's output led to the given score, especially if below threshold."
    )
    feedback: str = OutputField(
        desc=(
            "Provide concrete and actionable feedback for the module to improve"
            " its output on similar inputs in the future. Focus on how to"
            " achieve a score >= the threshold. If the module met the"
            " threshold, write N/A."
        )
    )


class Refine(Module):
    """
    Enhances prediction quality through iterative refinement based on a user-provided reward function.

    This module takes a base DSPy module (defined by its `signature`) and attempts to improve its outputs
    over `N` iterations using a reward mechanism controlled by `reward_fn` and `threshold`.

    The `reward_fn` is crucial and **must** perform the following:
    - Accept two arguments: `args` (the original keyword arguments passed to `Refine.forward`) and
      `pred` (the `dspy.Prediction` generated by the internal module in the current iteration).
    - Return a `dspy.Prediction` object. This returned object **must** contain at least two fields:
        - `score` (float or convertible to float): A numerical score indicating the quality of the input `pred`.
        - `feedback` (str): Actionable feedback for the underlying module on how to improve its
          next attempt, especially if the score is below the `threshold`. This feedback will be included
          in the `previous_attempts` context for the next iteration.

    Implementing Programmatic Constraints within `reward_fn`:
    Users can implement basic programmatic checks (like word count, format validation, etc.)
    If these checks fail, the `reward_fn` can return a `dspy.Prediction` with a low score (e.g., 0.0)
    and feedback detailing the constraint failures. The example below demonstrates this pattern.

    The refinement process continues until a prediction achieves a score greater than or equal to the
    `threshold` (after potentially passing internal programmatic checks within `reward_fn`),
    or until `N` iterations are completed. The internal predictor module receives the full
    history of previous outputs, scores, and feedback via the `previous_attempts` input field.

    Example:
    >>> import dspy
    >>>
    >>> # Define a Signature for the base task
    >>> class GenerateQA(dspy.Signature):
    ...     "Answer a question."
    ...     question = dspy.InputField()
    ...     answer = dspy.OutputField(desc="A concise answer.")
    >>>
    >>> # Define a Signature for assessing quality (used *after* programmatic checks pass)
    >>> class AssessQuality(dspy.Signature):
    ...     "Assess the quality of the generated answer."
    ...     question = dspy.InputField()
    ...     answer = dspy.InputField()
    ...     assessment_score = dspy.OutputField(desc="A score from 0.0 to 1.0.")
    ...     assessment_feedback = dspy.OutputField(desc="Feedback to improve the answer.")
    >>>
    >>> # Define the reward function - with internal programmatic checks
    >>> def assess_quality_with_constraints(args, pred):
    ...     MIN_WORDS = 5
    ...     MAX_WORDS = 50
    ...     TARGET_FIELD = "answer"
    ...     constraint_failures = []
    ...     output_value = getattr(pred, TARGET_FIELD, "")
    ...
    ...     # Programmatic check for word count
    ...     if isinstance(output_value, str):
    ...         word_count = len(output_value.split())
    ...         if word_count <= MIN_WORDS:
    ...             constraint_failures.append(f"Constraint Failed: '{TARGET_FIELD}' is too short ({word_count} < {MIN_WORDS} words).")
    ...         elif word_count >= MAX_WORDS:
    ...             constraint_failures.append(f"Constraint Failed: '{TARGET_FIELD}' is too long ({word_count} > {MAX_WORDS} words).")
    ...     else:
    ...         constraint_failures.append(f"Constraint Check Error: Field '{TARGET_FIELD}' is not a string.")
    ...
    ...     # Return failure if constraints were violated
    ...     if constraint_failures:
    ...         feedback_str = "Issues with the response:\\n"
    ...         for failure_msg in constraint_failures:
    ...             feedback_str += f"\\t* {failure_msg}\\n"
    ...         return dspy.Prediction(score=0.0, feedback=feedback_str)
    ...
    ...     # Constraints passed - proceed to LLM-based assessment
    ...     question = args.get("question", "")
    ...     assess_predictor = dspy.Predict(AssessQuality)
    ...     assessment_result = assess_predictor(question=question, answer=output_value)
    ...     # Ensure score is treated as float, handle potential conversion errors robustly if needed
    ...     try:
    ...         score = float(assessment_result.assessment_score)
    ...     except (ValueError, TypeError):
    ...         print(f"Warning: Could not convert assessment_score '{assessment_result.assessment_score}' to float. Defaulting to 0.0.")
    ...         score = 0.0 # Default score if conversion fails
    ...     feedback = assessment_result.assessment_feedback
    ...     return dspy.Prediction(score=score, feedback=feedback)
    >>>
    >>> qa_module = dspy.Predict(GenerateQA)
    >>> refined_qa = Refine(
    ...     signature=qa_module.signature,
    ...     reward_fn=assess_quality_with_constraints,
    ...     threshold=0.8,
    ...     N=3,
    ...     verbose=False
    ... )
    >>>
    >>> # --- Hypothetical Run ---
    >>> # Assume some LM is configured, e.g., dspy.settings.configure(lm=YourLanguageModel())
    >>> # Assume question = "What are mitochondria?"
    >>>
    >>> # Iteration 1:
    >>> # predictor generates: answer="Powerhouse." (1 word)
    >>> # reward_fn checks constraints: Fails MIN_WORDS (1 < 5).
    >>> # reward_fn returns: Prediction(score=0.0, feedback="Issues with the response: Constraint Failed: 'answer' is too short (1 < 5 words).")
    >>> # >> Content-based assessment (LLM call to AssessQuality) is skipped due to constraint failure.
    >>>
    >>> # Iteration 2:
    >>> # predictor receives previous attempt info and generates: answer="Mitochondria are the powerhouse of the cell." (7 words)
    >>> # reward_fn checks constraints: Passes MIN_WORDS and MAX_WORDS.
    >>> # reward_fn proceeds to content assessment by calling the LLM via dspy.Predict(AssessQuality).
    >>> # (Hypothetical LLM assessment returns score=0.9, feedback="Good answer! Concise and captures the key function, though could add that mitochondria are organelles in eukaryotic cells with their own DNA.")
    >>> # reward_fn returns: Prediction(score=0.9, feedback="Good answer! Concise and captures the key function, though could add that mitochondria are organelles in eukaryotic cells with their own DNA.")
    >>> # Success! Score 0.9 >= threshold 0.8. Refinement stops.
    >>>
    >>> # Now, calling refined_qa executes this process:
    >>> # final_result = refined_qa(question="What are mitochondria?")
    >>> # The final prediction stored in final_result would contain the successful attempt:
    >>> # print(final_result.answer)
    >>> # Output would likely be: "Mitochondria are the powerhouse of the cell."
    """
    def __init__(
        self,
        signature: Signature,
        reward_fn: Callable[[dict[str, Any], Prediction], Prediction],
        threshold: float = 0.0,
        N: int = 3,
        use_cot: bool = False,
        verbose: bool = False,
        fail_if_below_threshold: bool = False,
        use_llm_feedback: bool = False
    ) -> None:
        """
        Initializes the Refine module.

        Args:
            signature (Signature): The DSPy signature defining inputs/outputs for the module being refined.
                                  The internal predictor module's signature will be this signature augmented
                                  with a `previous_attempts` input field.
            reward_fn (Callable): A function that evaluates a generated prediction in the context of the
                                 original inputs. It must return a `dspy.Prediction` object containing
                                 'score' (float or convertible) and 'feedback' (str) attributes.
                                 Function signature: fn(args: dict[str, Any], pred: Prediction) -> Prediction
            threshold (float): Minimum acceptable score. Refinement stops if this score is met or exceeded.
                              Defaults to 0.7.
            N (int): Maximum number of refinement iterations to attempt. Defaults to 3.
            use_cot (bool): If True, uses `dspy.ChainOfThought` for the internal predictor module,
                           encouraging step-by-step reasoning. If False, uses `dspy.Predict`.
                           Defaults to False.
            verbose (bool): If True, prints detailed logs of the refinement process, including
                           intermediate outputs, scores, and feedback. Defaults to False.
            fail_if_below_threshold (bool): If True, raises a `ValueError` if the `threshold` score
                                           is not met after `N` iterations. If False, returns the
                                           best prediction found even if it's below the threshold.
                                           Defaults to False.
            use_llm_feedback (bool): If True, attempts to use the `OfferFeedback` signature to
                                    generate feedback for the predictor module instead of using
                                    the feedback string directly from `reward_fn`. The `reward_fn`
                                    is still required for scoring. Defaults to False.
        """
        super().__init__()
        assert signature is not None, "A `signature` must be provided to Refine."
        assert reward_fn is not None, "`reward_fn` must be provided."
        assert isinstance(
            threshold, (float, int)
        ), "`threshold` must be a numerical value."

        self.reward_fn = reward_fn
        self.threshold = threshold
        self.N = max(1, N)
        self.use_cot = use_cot
        self.verbose = verbose
        self.fail_on_threshold = fail_if_below_threshold
        self.base_signature: Signature = copy.deepcopy(signature)
        self.use_llm_feedback = use_llm_feedback

        predictor_signature: Signature = copy.deepcopy(signature)
        if "previous_attempts" not in predictor_signature.input_fields:
            predictor_signature = predictor_signature.append(
                "previous_attempts",
                dspy.InputField(desc=f"The history of previous attempts towards producing {list(predictor_signature.output_fields.keys())} given {list(predictor_signature.input_fields.keys())}, and associated feedback for each attempt"),
            )
        self.predictor_signature: Signature = predictor_signature
        predictor_class: type[Module] = (dspy.ChainOfThought if self.use_cot else dspy.Predict)
        self.predictor_module: Module = predictor_class(self.predictor_signature)

        # Code Inspection
        self.module_code = inspect.getsource(self.predictor_module.__class__)
        try:
            self.reward_fn_code = inspect.getsource(reward_fn) if reward_fn else ""
        except (TypeError, OSError):
            self.reward_fn_code = ""

    def _get_temperatures(self) -> list[float]:
        """
        Generate a sequence of temperatures for multiple iterations.

        Returns:
            list: A list of temperature values to use for each iteration.
        """
        lm = dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        return temps

    @staticmethod
    def _format_previous_attempts(attempts_log: list[dict[str, Any]]) -> str:
        """
        Formats the log of previous refinement attempts into a multi-line string.

        This string is suitable for inclusion as input context (`previous_attempts`)
        to the internal predictor module, allowing it to learn from past mistakes and feedback.

        Args:
            attempts_log (list[dict[str, Any]]): A list where each dictionary represents
                a past attempt, potentially containing 'attempt_number', 'output',
                'score', 'feedback', and 'error'.

        Returns:
            str: A formatted string detailing previous attempts, or "N/A" if the log is empty.

        Example:
            >>> attempts_log = [
            ...     {
            ...         "attempt_number": 1,
            ...         "output": {"answer": "They make energy."},
            ...         "score": 0.6,
            ...         "feedback": "Be more specific about the type of energy produced."
            ...     },
            ...     {
            ...         "attempt_number": 2,
            ...         "output": {"answer": "They make ATP energy."},
            ...         "score": 0.7,
            ...         "feedback": "Mention 'cellular respiration' in your answer."
            ...     }
            ... ]
            >>> formatted = Refine._format_previous_attempts(attempts_log)
            >>> print(formatted)
            --- PREVIOUS ATTEMPTS HISTORY ---
            ATTEMPT 1:
              Output:
                answer: They make energy.
              Score: 0.6
              Feedback Given After This Attempt: Be more specific about the type of energy produced.
            ----------
            ATTEMPT 2:
              Output:
                answer: They make ATP energy.
              Score: 0.7
              Feedback Given After This Attempt: Mention 'cellular respiration' in your answer.
            ----------
            --- END PREVIOUS ATTEMPTS HISTORY ---
        """
        if not attempts_log:
            return "N/A"
        formatted_parts: list[str] = ["--- PREVIOUS ATTEMPTS HISTORY ---"]
        for idx, attempt in enumerate(attempts_log):
            attempt_num: int = attempt.get("attempt_number", idx + 1)
            formatted_parts.append(f"ATTEMPT {attempt_num}:")
            if "output" in attempt:
                outputs: dict[str, Any] = attempt.get("output", {})
                formatted_parts.append("  Output:")
                for key, value in outputs.items():
                    value_str: str = str(value).replace("\n", "\n    ")
                    formatted_parts.append(f"    {key}: {value_str}")
            if "score" in attempt:
                score_val = attempt.get("score", "N/A")
                formatted_parts.append(f"  Score: {score_val}")
            feedback: Union[str, None] = attempt.get("feedback")
            if feedback and feedback != "N/A":
                feedback_str: str = str(feedback).replace("\n", "\n    ")
                formatted_parts.append(f"  Feedback Given After This Attempt: {feedback_str}")
            else:
                formatted_parts.append("  Feedback Given After This Attempt: N/A")
            if "error" in attempt:
                error_str: str = str(attempt["error"]).replace("\n", "\n    ")
                formatted_parts.append(f"  Error During This Attempt: {error_str}")
            formatted_parts.append("-" * 10)
        formatted_parts.append("--- END PREVIOUS ATTEMPTS HISTORY ---")
        return "\n".join(formatted_parts)

    def _generate_llm_feedback(self, kwargs: dict[str, Any], trace: list[Any], output_dict: dict[str, Any], current_score: float) -> str:
        """
        Generates feedback for a module using the `OfferFeedback` signature.

        This helper method encapsulates the process of calling the OfferFeedback module
        to generate actionable feedback for the predictor module based on its performance.

        Args:
            kwargs (dict[str, Any]): The original input arguments to the module.
            trace (list[Any]): The execution trace of the module.
            output_dict (dict[str, Any]): The outputs generated by the module.
            current_score (float): The reward score assigned to the module's output.

        Returns:
            str: The generated feedback for improving the module's performance.
        """
        feedback_module = dspy.Predict(OfferFeedback)
        feedback_inputs = {
            "program_code": self.module_code,
            "modules_defn": inspect_modules(self.predictor_module),
            "program_inputs": str(kwargs),
            "program_trajectory": trace,
            "program_outputs": output_dict,
            "reward_code": self.reward_fn_code,
            "target_threshold": self.threshold,
            "reward_value": current_score
        }
        feedback_result = feedback_module(**feedback_inputs)
        return feedback_result.feedback

    def forward(self, **kwargs: dict) -> Prediction:
        """
        Executes the iterative refinement process.

        Takes keyword arguments matching the input fields of the provided `signature`. It then
        repeatedly calls the internal predictor module (providing the history via `previous_attempts`),
        evaluates the output using `reward_fn`, extracts score and feedback from the `reward_fn`'s result,
        and logs this information until the `threshold` is met or `N` iterations are completed.

        Args:
            **kwargs (dict): Keyword arguments corresponding to the input fields defined in the
                            `signature` passed during initialization.

        Returns:
            Prediction: The prediction object corresponding to the attempt with the highest score
                        achieved during the refinement process. This object will have an additional
                        attribute `Refine_metadata` (dict) containing details about the refinement run
                        (iterations, final score, success status, attempt logs).

        Raises:
            ValueError: If `fail_on_threshold` is True and the best score achieved after N iterations
                        is still below the `threshold`.
            RuntimeError: If the module fails to produce any valid prediction across all N attempts
                          (e.g., due to repeated LLM errors or exceptions in `reward_fn`).
            TypeError: If the provided `reward_fn` does not return a `dspy.Prediction` object, or if
                       that object lacks the required 'score' or 'feedback' attributes, or if 'score'
                       is not convertible to float.
        """
        lm = dspy.settings.lm
        temps: list[float] = self._get_temperatures()
        iterations_made: int = 0
        best_prediction: Union[Prediction, None] = None
        best_score: float = -float("inf")
        feedback_leading_to_best_prediction: Union[str, None] = None
        attempts_log: list[dict[str, Any]] = []
        best_trace: list[Any] | None = None
        if self.verbose:
            print(f"Starting Refine with N={self.N}, threshold={self.threshold}")
        for idx, temp in enumerate(temps):
            iterations_made = idx + 1
            current_attempt_info: dict[str, Any] = {
                "attempt_number": iterations_made
            }
            if self.verbose:
                print(f"\nIteration {iterations_made}/{self.N} with temperature {temp}")
            current_kwargs: dict[str, Any] = kwargs.copy()
            current_kwargs["previous_attempts"] = self._format_previous_attempts(attempts_log)
            lm_temp: Union[Any, None] = None
            if lm:
                try:
                    lm_temp = lm.copy(temperature=temp)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error setting temperature {temp}: {e}. Using original LM settings.")
                    lm_temp = lm
            try:
                with dspy.context(lm=lm_temp, trace=[]):
                    outputs = self.predictor_module(**current_kwargs)
                    trace = dspy.settings.trace
                output_dict: dict[str, Any] = {k: outputs[k] for k in self.base_signature.output_fields}
                current_attempt_info["output"] = output_dict
                if self.verbose:
                    print("Generated Output:")
                    for k, v in output_dict.items():
                        # Replace newlines with newline + 4 spaces to ensure multi-line values are indented consistently below their key
                        print(f"  {k}: {str(v).replace(chr(10), chr(10) + '    ')}")
                reward_result: Prediction = self.reward_fn(kwargs, outputs)
                if not isinstance(reward_result, Prediction):
                    raise TypeError(f"`reward_fn` must return a `dspy.Prediction` object Got type: {type(reward_result)}")
                if not (hasattr(reward_result, "score") and hasattr(reward_result, "feedback")):
                    raise TypeError("The `dspy.Prediction` returned by `reward_fn` must have 'score' and 'feedback' attributes.")
                try:
                    current_score = float(reward_result.score)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        "The 'score' attribute from reward_fn's Prediction must"
                        " be convertible to float. Error:"
                        f" {e}. Got score:"
                        f" {getattr(reward_result, 'score', 'MISSING')}"
                    ) from e
                current_feedback = str(reward_result.feedback)
                if self.use_llm_feedback and current_score < self.threshold:
                    llm_feedback = self._generate_llm_feedback(
                        kwargs=kwargs,
                        trace=trace,
                        output_dict=output_dict,
                        current_score=current_score
                    )
                    current_feedback += f"\n\n[LLM Refinement Feedback]:\n{llm_feedback}"
                if self.verbose:
                    print(f"Reward function returned: score={current_score}, feedback='{current_feedback}'")
                current_attempt_info["score"] = current_score
                current_attempt_info["feedback"] = current_feedback
                if current_score > best_score:
                    if self.verbose:
                        print(f"Found new best prediction (Score: {current_score})")
                    best_prediction = outputs
                    best_trace = trace
                    best_score = current_score
                    feedback_leading_to_best_prediction = current_feedback
                success: bool = current_score >= self.threshold
                attempts_log.append(current_attempt_info)
                if success:
                    if self.verbose:
                        print(
                            f"✓ Success: Score {current_score} meets threshold"
                            f" {self.threshold} after {iterations_made}"
                            " iterations."
                        )
                    break
                else:
                    if self.verbose:
                        print(f"Score {current_score} is below threshold: {self.threshold}. Continuing refinement.")
            except Exception as e:
                if self.verbose:
                    print(
                        f"✗ Iteration {iterations_made} failed with exception:"
                        f" {type(e).__name__}: {e}"
                    )
                current_attempt_info["error"] = f"{type(e).__name__}: {e}"
                attempts_log.append(current_attempt_info)
        if best_prediction is None:
            raise RuntimeError(
                f"Refine failed to produce any valid prediction after {self.N}"
                " attempts. Check logs for errors."
            )
        refine_successful: bool = best_score >= self.threshold
        metadata: dict[str, Any] = {
            "iterations_made": iterations_made,
            "N": self.N,
            "best_score": best_score,
            "feedback_leading_to_best_prediction": feedback_leading_to_best_prediction,
            "target_threshold": self.threshold,
            "refine_successful": refine_successful,
            "attempts_log": attempts_log,
        }
        setattr(best_prediction, "Refine_metadata", metadata)
        if self.fail_on_threshold and not refine_successful:
            error_message: str = (
                f"Refine failed to meet the score threshold of {self.threshold}"
                f" after {iterations_made} iterations. Best score achieved:"
                f" {best_score}."
            )
            raise ValueError(error_message)
        if best_trace:
            dspy.settings.trace.clear()
            dspy.settings.trace.extend(best_trace)
        return best_prediction


def inspect_modules(program):
    """
    Generates a formatted string representation of modules in a DSPy program.

    This function examines a DSPy program and creates a detailed textual description
    of each module it contains, including input fields, output fields, and instructions.
    The output is formatted with clear separators between modules for improved readability.

    Args:
        program: A DSPy module or program containing named predictors to be inspected.

    Returns:
        str: A formatted multi-line string containing detailed descriptions of all modules
             in the program, with separator lines between each module's information.
    """
    separator = "-" * 80
    output = [separator]
    for idx, (name, predictor) in enumerate(program.named_predictors()):
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())
        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)
    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o: Any) -> Any:
    """
    Recursively mask non-serializable objects to make them JSON-serializable.

    Args:
        o: Any Python object.

    Returns:
        A JSON-serializable version of the input object.
    """
    try:
        ujson.dumps(o)
        return o
    except TypeError:
        pass
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    else:
        return f"<non-serializable: {type(o).__name__}>"