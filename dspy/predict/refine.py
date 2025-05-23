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
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    reward_code: str = InputField(desc="The code of the reward function that we are analyzing")
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
        "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
        "scenario in the future. If the module is not to blame, write N/A."
    )


def format_llm_feedback(advice_dict: dict[str, str] | None) -> str:
    """
    Formats OfferFeedback output into a readable string for previous_attempts field.

    Example:
    ```python
    feedback = {
        "AnswerGenerator": "Include more specific details from the retrieved context in your responses.",
        "FactChecker": "N/A"
    }
    print(format_llm_feedback(feedback))
    # Output:
    # MODULE-SPECIFIC FEEDBACK:
    # MODULE: AnswerGenerator
    #   Advice: Include more specific details from the retrieved context in your responses.
    # MODULE: FactChecker
    #   Advice: No specific improvements needed for this module.
    ```
    """
    if not isinstance(advice_dict, dict):
        return f"Expected dictionary of advice, got {type(advice_dict).__name__}"
    formatted_parts: list[str] = ["MODULE-SPECIFIC FEEDBACK:"]
    for module_name, advice in sorted(advice_dict.items()):
        formatted_parts.append(f"MODULE: {module_name}")
        if advice.strip().upper() == "N/A" or not advice.strip():
            formatted_parts.append("\tAdvice: No specific improvements needed for this module.")
        else:
            advice_lines: list[str] = advice.strip().split("\n")
            if len(advice_lines) == 1:
                formatted_parts.append(f"\tAdvice: {advice_lines[0]}")
            else:
                formatted_parts.append("\tAdvice:")
                for line in advice_lines:
                    formatted_parts.append(f"\t\t{line}")
    return "\n".join(formatted_parts)


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
    ```python
    import dspy

    # Define a Signature for the base task
    class GenerateQA(dspy.Signature):
        "Answer a question."
        question = dspy.InputField()
        answer = dspy.OutputField(desc="A concise answer.")

    # Define a Signature for assessing quality (used *after* programmatic checks pass)
    class AssessQuality(dspy.Signature):
        "Assess the quality of the generated answer."
        question = dspy.InputField()
        answer = dspy.InputField()
        assessment_score = dspy.OutputField(desc="A score from 0.0 to 1.0.")
        assessment_feedback = dspy.OutputField(desc="Feedback to improve the answer.")

    # Define the reward function - with internal programmatic checks
    def assess_quality_with_constraints(args, pred):
        MIN_WORDS = 5
        MAX_WORDS = 50
        TARGET_FIELD = "answer"
        constraint_failures = []
        output_value = getattr(pred, TARGET_FIELD, "")

        # Programmatic check for word count
        if isinstance(output_value, str):
            word_count = len(output_value.split())
            if word_count <= MIN_WORDS:
                constraint_failures.append(f"Constraint Failed: '{TARGET_FIELD}' is too short ({word_count} < {MIN_WORDS} words).")
            elif word_count >= MAX_WORDS:
                constraint_failures.append(f"Constraint Failed: '{TARGET_FIELD}' is too long ({word_count} > {MAX_WORDS} words).")
        else:
            constraint_failures.append(f"Constraint Check Error: Field '{TARGET_FIELD}' is not a string.")

        # Return failure if constraints were violated
        if constraint_failures:
            feedback_str = "Issues with the response:\n"
            for failure_msg in constraint_failures:
                feedback_str += f"\t* {failure_msg}\n"
            return dspy.Prediction(score=0.0, feedback=feedback_str)

        # Constraints passed - proceed to LLM-based assessment
        question = args.get("question", "")
        assess_predictor = dspy.Predict(AssessQuality)
        assessment_result = assess_predictor(question=question, answer=output_value)
        # Ensure score is treated as float, handle potential conversion errors robustly if needed
        try:
            score = float(assessment_result.assessment_score)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert assessment_score '{assessment_result.assessment_score}' to float. Defaulting to 0.0.")
            score = 0.0 # Default score if conversion fails
        feedback = assessment_result.assessment_feedback
        return dspy.Prediction(score=score, feedback=feedback)

    qa_module = dspy.Predict(GenerateQA)
    refined_qa = Refine(
        signature=qa_module.signature,
        reward_fn=assess_quality_with_constraints,
        threshold=0.8,
        N=3,
        verbose=False
    )

    # When using the refined_qa module:
    result = refined_qa(question="What are mitochondria?")
    print(result.answer)  # Will contain the best answer after refinement
    ```
    """
    def __init__(
        self,
        signature: Signature,
        reward_fn: Callable[[dict[str, Any], Prediction], Prediction],
        threshold: Union[float, bool, int] = 0.0,
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
                The internal predictor module's signature will be this signature augmented with a `previous_attempts` input field.
            reward_fn (Callable): A function that evaluates a generated prediction in the context of the
                original inputs. It must return a `dspy.Prediction` object containing 'score' (float or convertible) and 'feedback' (str) attributes.
                Function signature: fn(args: dict[str, Any], pred: Prediction) -> Prediction
            threshold (Union[float, bool, int]): Minimum acceptable score. Refinement stops if this score is met or exceeded.
                Can be a float, boolean, or integer value. If a boolean is provided, True is treated as 1.0 and False as 0.0.
                Defaults to 0.7.
            N (int): Maximum number of refinement iterations to attempt. Defaults to 3.
            use_cot (bool): If True, uses `dspy.ChainOfThought` for the internal predictor module,
                encouraging step-by-step reasoning. If False, uses `dspy.Predict`. Defaults to False.
            verbose (bool): If True, prints detailed logs of the refinement process, including
                intermediate outputs, scores, and feedback. Defaults to False.
            fail_if_below_threshold (bool): If True, raises a `ValueError` if the `threshold` score
                is not met after `N` iterations. If False, returns the best prediction found even if it's below the threshold.
                Defaults to False.
            use_llm_feedback (bool): If True, attempts to use the `OfferFeedback` signature to
                generate feedback for the predictor module instead of using the feedback string directly from `reward_fn`.
                The `reward_fn` is still required for scoring. Defaults to False.
        """
        super().__init__()
        assert signature is not None, "A `signature` must be provided to Refine."
        assert reward_fn is not None, "`reward_fn` must be provided."
        assert isinstance(threshold, (float, int, bool)), "`threshold` must be a numerical/bool value."

        self.reward_fn = reward_fn
        self.threshold = float(threshold)
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
                a past attempt, potentially containing 'attempt_number', 'output', 'score', 'feedback', and 'error'.

        Returns:
            str: A formatted string detailing previous attempts, or "N/A" if the log is empty.

        Example:
        ```python
        attempts_log = [
            {
                "attempt_number": 1,
                "output": {"answer": "They make energy."},
                "score": 0.6,
                "feedback": "Be more specific about the type of energy produced."
            },
            {
                "attempt_number": 2,
                "output": {"answer": "They make ATP energy."},
                "score": 0.7,
                "feedback": "Mention 'cellular respiration' in your answer."
            }
        ]
        formatted = Refine._format_previous_attempts(attempts_log)
        print(formatted)
        # Output:
        # --- PREVIOUS ATTEMPTS HISTORY ---
        # ATTEMPT 1:
        #   Output:
        #     answer: They make energy.
        #   Score: 0.6
        #   Feedback Given After This Attempt: Be more specific about the type of energy produced.
        # ----------
        # ATTEMPT 2:
        #   Output:
        #     answer: They make ATP energy.
        #   Score: 0.7
        #   Feedback Given After This Attempt: Mention 'cellular respiration' in your answer.
        # ----------
        # --- END PREVIOUS ATTEMPTS HISTORY ---
        ```
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

    def _generate_advice_dict(
        self,
        program_inputs_dict: dict[str, Any],
        trace: list[Any],
        outputs_dict: dict[str, Any],
        current_score: float,
        predictor2name: dict[Module, str],
    ) -> dict[str, str]:
        """
        Generates the structured advice dictionary using the `OfferFeedback` signature.

        This helper method is called when a refinement attempt's score is below the
        threshold and LLM feedback is enabled. It prepares the necessary context
        (program code, module definitions, inputs, execution trajectory, outputs,
        reward details) and calls the `OfferFeedback` module. The goal is to generate
        concrete, actionable advice targeted at specific sub-modules (identified via
        `predictor2name`) based on their performance in the current trace. This advice
        dictionary is then used for hint injection in the subsequent refinement attempt.

        Args:
            program_inputs_dict (dict[str, Any]): The original keyword arguments passed to
                the `Refine.forward` method for this run.
            trace (list[Any]): The execution trace captured during the `predictor_module`'s
                run for the current attempt. Typically, contains tuples or objects representing predictor calls, their inputs, and outputs.
            outputs_dict (dict[str, Any]): The primary outputs generated by the `predictor_module` for the current attempt.
            current_score (float): The reward score assigned by the `reward_fn` to the current attempt's output.
            predictor2name (dict[Module, str]): A mapping from sub-module instances within
                the `predictor_module` to their assigned string names.
                Used for structuring the trajectory context and potentially enabling targeted advice retrieval later.
        Returns:
            dict[str, str]: A dictionary mapping module names (str) to specific advice/hint
                strings (str) generated by `OfferFeedback`. Returns a dictionary
                containing an 'error' key if the feedback generation process fails
                or if the returned format is invalid.
        """
        if self.verbose:
            print("Generating LLM feedback/advice...")
        module_definition = inspect_modules(self.predictor_module)
        module_names = list(predictor2name.values())
        modules_context = dict(program_code=self.module_code, modules_defn=module_definition)
        try:
             trajectory_list = []
             for predictor_instance, inputs_obj, outputs_obj in trace:
                 module_name = predictor2name.get(predictor_instance, "unknown_predictor")
                 inputs_dict = inputs_obj.toDict() if hasattr(inputs_obj, 'toDict') else dict(inputs_obj) if isinstance(inputs_obj, dict) else {"input_value": repr(inputs_obj)}
                 outputs_formatted = dict(outputs_obj) if hasattr(outputs_obj, 'items') else {"output_value": repr(outputs_obj)}
                 trajectory_list.append(dict(module_name=module_name, inputs=inputs_dict, outputs=outputs_formatted))
        except Exception as e:
             print(f"Warning: Error processing trace for feedback generation: {type(e).__name__}: {e}")
             trajectory_list = [{"error": f"Failed to process trace: {type(e).__name__}"}]
        trajectory_context = dict(
            program_inputs=program_inputs_dict,
            program_trajectory=trajectory_list,
            program_outputs=outputs_dict
        )
        reward_context = dict(
            reward_code=self.reward_fn_code,
            target_threshold=self.threshold,
            reward_value=current_score
        )
        advise_kwargs = dict(**modules_context, **trajectory_context, **reward_context, module_names=module_names)
        serialized_advise_kwargs = {}
        for k, v in advise_kwargs.items():
            if isinstance(v, (dict, list, tuple)):
                 try:
                     masked_v = recursive_mask(v)
                     serialized_advise_kwargs[k] = ujson.dumps(masked_v, indent=2)
                 except Exception as dump_error:
                     print(f"Warning: Could not serialize key '{k}' for OfferFeedback: {dump_error}. Using repr.")
                     serialized_advise_kwargs[k] = repr(v)
            elif isinstance(v, str):
                serialized_advise_kwargs[k] = v
            else:
                 try:
                     serialized_advise_kwargs[k] = str(v)
                 except Exception:
                     serialized_advise_kwargs[k] = repr(v)
        try:
            feedback_module = dspy.ChainOfThought(OfferFeedback)
            feedback_result = feedback_module(**serialized_advise_kwargs)
            advice_dict = feedback_result.advice
            if not isinstance(advice_dict, dict):
                print(f"Warning: OfferFeedback returned advice that wasn't a dictionary. Type: {type(advice_dict)}. Value: {advice_dict}")
                return {"error": f"Invalid advice format: Expected dict, got {type(advice_dict)}"}
            if self.verbose:
                 print(f"Received advice: {ujson.dumps(advice_dict, indent=2)}")
            return advice_dict
        except Exception as e:
            print(f"Error during LLM feedback generation: {e}")

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
                that object lacks the required 'score' or 'feedback' attributes, or if 'score' is not convertible to float.
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
                    predictor2name = {}
                    for predictor_instance, _, _ in trace:
                        predictor2name = {predictor: name for name, predictor in self.predictor_module.named_predictors()}
                    llm_feedback_dict = self._generate_advice_dict(
                        program_inputs_dict=kwargs,
                        trace=trace,
                        outputs_dict=output_dict,
                        current_score=current_score,
                        predictor2name=predictor2name
                    )
                    formatted_llm_feedback = format_llm_feedback(llm_feedback_dict)
                    current_feedback += f"\n\n{formatted_llm_feedback}"
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