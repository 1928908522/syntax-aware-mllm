@echo off
python -m syntax_visual_router.data.extract_triples --input data\captions.jsonl --output data\ranked_triples.jsonl
python -m syntax_visual_router.data.build_candidates --input data\ranked_triples.jsonl --output data\perturbation_candidates.jsonl
python -m syntax_visual_router.data.generate_negatives --input data\perturbation_candidates.jsonl --output_jsonl data\generated_negatives.jsonl --triple_mode topk --top_k 1
python -m syntax_visual_router.data.filter_negatives --input data\generated_negatives.jsonl --output data\syntax_negatives.jsonl
python -m syntax_visual_router.data.build_cot_data --input_jsonl data\syntax_negatives.jsonl --out_dir data
python -m syntax_visual_router.data.build_preference_data --input_jsonl data\stage_e_cot_sft_topk_train.jsonl --output_jsonl data\stage_e_preference_topk_train.jsonl
python -m syntax_visual_router.data.build_syntax_rl_data --input_jsonl data\stage_e_cot_sft_topk_train.jsonl --output_jsonl data\stage_e_syntax_rl_train.jsonl
python -m syntax_visual_router.data.build_syntax_rl_data --input_jsonl data\stage_e_cot_sft_topk_test.jsonl --output_jsonl data\stage_e_syntax_rl_test.jsonl
