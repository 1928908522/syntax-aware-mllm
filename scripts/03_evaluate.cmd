@echo off
python -m syntax_visual_router.evaluation.evaluate_reasoning --adapter checkpoints\cot_sft\stage_e1_epoch01.pt --input_jsonl data\stage_e_syntax_rl_test.jsonl --output_jsonl outputs\cot_sft_test.jsonl
python -m syntax_visual_router.evaluation.evaluate_reasoning --adapter checkpoints\syntax_dpo\stage_e2_dpo_epoch01.pt --input_jsonl data\stage_e_syntax_rl_test.jsonl --output_jsonl outputs\dpo_test.jsonl
python -m syntax_visual_router.evaluation.evaluate_reasoning --adapter checkpoints\syntax_rl\stage_e2_syntax_rl_epoch01.pt --input_jsonl data\stage_e_syntax_rl_test.jsonl --output_jsonl outputs\syntax_rl_test.jsonl
