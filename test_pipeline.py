import unittest
from unittest.mock import patch

from pipeline import PolicyChunk, answer_policy_question


class AnswerPolicyQuestionTests(unittest.TestCase):
    def test_retries_after_malformed_gemini_json(self) -> None:
        chunk = PolicyChunk(
            chunk_id="chunk-1",
            policy_id="book-1",
            title="Example textbook",
            section="Definitions",
            page=1,
            text="A parameter is a quantity that describes a population.",
        )
        valid_response = {
            "answer_found": True,
            "conflict_detected": False,
            "answer": "A parameter describes a population.",
            "citations": [],
            "confidence": "high",
            "escalation_note": None,
        }

        with (
            patch("pipeline.embed_query", return_value=[0.0]),
            patch("pipeline.vector_search", return_value=[chunk]),
            patch("pipeline.rerank", return_value=[chunk]),
            patch(
                "pipeline.call_gemini",
                side_effect=[
                    ValueError("Malformed JSON from model"),
                    valid_response,
                ],
            ) as call_gemini,
            patch("pipeline.time.sleep") as sleep,
        ):
            response = answer_policy_question("What are parameters?", object())

        self.assertTrue(response.answer_found)
        self.assertEqual(response.answer, valid_response["answer"])
        self.assertEqual(call_gemini.call_count, 2)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
