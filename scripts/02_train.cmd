@echo off
python -m syntax_visual_router.training.train_cot_sft --input_jsonl data\stage_e_cot_sft_topk_train.jsonl --output_dir checkpoints\cot_sft
python -m syntax_visual_router.training.precompute_dpo_reference --input_jsonl data\stage_e_preference_topk_train.jsonl --adapter checkpoints\cot_sft\stage_e1_epoch01.pt --output_jsonl data\stage_e_preference_topk_train_ref_mean.jsonl
python -m syntax_visual_router.training.train_dpo --input_jsonl data\stage_e_preference_topk_train_ref_mean.jsonl --init_adapter checkpoints\cot_sft\stage_e1_epoch01.pt --reference_adapter checkpoints\cot_sft\stage_e1_epoch01.pt --use_precomputed_ref --output_dir checkpoints\syntax_dpo
python -m syntax_visual_router.training.train_syntax_rl --input_jsonl data\stage_e_syntax_rl_train.jsonl --init_adapter checkpoints\syntax_dpo\stage_e2_dpo_epoch01.pt --output_dir checkpoints\syntax_rl --route_sample_weight_coef 0
