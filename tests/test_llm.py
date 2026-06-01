import pytest
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric

# 1. Define your metrics thresholds (e.g., max 5% error means > 0.95 score)
hallucination_metric = HallucinationMetric(threshold=0.5) # Adjust based on strictness
relevancy_metric = AnswerRelevancyMetric(threshold=0.8)
faithfulness_metric = FaithfulnessMetric(threshold=0.8)

# 2. Load your golden dataset (Looping through your 100 QA pairs)
@pytest.mark.parametrize("test_data", load_golden_dataset())
def test_llm_pipeline(test_data):
    # Simulate your RAG system / Prompt call
    start_time = time.time()
    
    # Replace this with your actual LLM / RAG invocation code
    actual_output = run_your_llm_pipeline(test_data["input"]) 
    
    latency = time.time() - start_time
    
    # Guardrail for Latency SLA (e.g., fail if > 4 seconds)
    assert latency < 4.0, f"Latency SLA breached: {latency}s"

    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
        retrieval_context=test_data["context"]
    )
    
    # Assert DeepEval metrics (Will automatically fail the pytest run if threshold isn't met)
    assert_test(test_case, [hallucination_metric, relevancy_metric, faithfulness_metric])