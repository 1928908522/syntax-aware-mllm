import unittest

from syntax_visual_router.training.rewards import (
    parse_key_relation,
    sample_confidence_weight,
    score_response,
    triple_similarity,
    target_triples,
)
from syntax_visual_router.data.build_syntax_rl_data import contrastive_triples


ROW = {
    "answer": "A",
    "selected_triple": {"src": "dog", "rel": "nsubj", "dst": "runs"},
    "target_triples": [{"src": "dog", "rel": "nsubj", "dst": "runs"}],
    "route_score": 0.1,
    "metrics": {"syntax_quality": 1.0, "nli_label": "contradiction"},
}


class StageERewardsTest(unittest.TestCase):
    def test_parse_and_exact_triple(self):
        parsed = parse_key_relation("Key relation: <Dog, nsubj, runs>.\nAnswer: A.")
        self.assertEqual(parsed, {"src": "dog", "rel": "nsubj", "dst": "runs"})
        self.assertAlmostEqual(triple_similarity(parsed, ROW["selected_triple"]), 1.0)

    def test_correct_answer_and_triple_receive_joint_reward(self):
        result = score_response(
            ROW,
            "Key relation: <dog, nsubj, runs>.\n"
            "Visual check: supported.\nContrast: differs.\nAnswer: A.",
        )
        self.assertEqual(result["answer"], 1.0)
        self.assertEqual(result["triple"], 1.0)
        self.assertEqual(result["joint"], 1.0)
        self.assertAlmostEqual(result["total"], 2.7)

    def test_partial_triple_is_scored_without_joint_bonus(self):
        result = score_response(
            ROW,
            "Key relation: <cat, nsubj, runs>.\n"
            "Visual check: supported.\nContrast: differs.\nAnswer: A.",
        )
        self.assertAlmostEqual(result["triple"], 0.65)
        self.assertEqual(result["joint"], 0.0)

    def test_missing_structure_is_penalized(self):
        result = score_response(ROW, "I cannot determine this.")
        self.assertEqual(result["total"], -1.0)
        self.assertEqual(result["penalty"], 1.0)

    def test_non_ascii_artifact_is_penalized(self):
        response = (
            "Key relation: <dog, nsubj, runs>.\n"
            "Visual check: supported.\nContrast: 混合 text.\nAnswer: A."
        )
        result = score_response(ROW, response)
        self.assertAlmostEqual(result["penalty"], 0.2)
        self.assertAlmostEqual(result["total"], 2.5)

    def test_route_has_no_effect_by_default(self):
        self.assertEqual(sample_confidence_weight(ROW, syntax_coef=0.25), 1.25)
        weighted = sample_confidence_weight(ROW, syntax_coef=0.25, route_coef=0.1)
        self.assertAlmostEqual(weighted, 1.21)

    def test_text_contrast_targets_do_not_use_route_score(self):
        positive = [
            {"src": "dog", "rel": "nsubj", "dst": "runs"},
            {"src": "red", "rel": "amod", "dst": "ball"},
        ]
        negative = [
            {"src": "dog", "rel": "nsubj", "dst": "runs"},
            {"src": "blue", "rel": "amod", "dst": "ball"},
        ]
        self.assertEqual(contrastive_triples(positive, negative), [positive[1]])

    def test_text_contrast_prefers_semantic_dependencies(self):
        positive = [
            {"src": "a", "rel": "det", "dst": "cloud"},
            {"src": "cloud", "rel": "pobj", "dst": "to"},
        ]
        self.assertEqual(contrastive_triples(positive, []), [positive[1]])

    def test_explicit_empty_targets_do_not_fall_back_to_route(self):
        row = {"target_triples": [], "selected_triple": ROW["selected_triple"]}
        self.assertEqual(target_triples(row), [])


if __name__ == "__main__":
    unittest.main()
