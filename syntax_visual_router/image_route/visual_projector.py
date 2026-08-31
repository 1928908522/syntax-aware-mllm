"""
模型模块: Qwen Visual Extractor

提取 Qwen2-VL Vision Encoder 输出的 V_raw [N, 1536]。
当前架构: 所有路由计算直接使用原始 V_raw，不做降维投影。
"""
import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVisualExtractor:
    """
    封装 Qwen2-VL 的视觉编码部分。
    只提取 V_raw，不涉及 LLM forward。
    """

    def __init__(self, model_path: str, device: str = "cuda", max_pixels: int = 313600):
        self.device = device
        self.max_pixels = max_pixels

        print(f"加载 Qwen2-VL 视觉编码器: {model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.image_processor.max_pixels = max_pixels

        # 冻结整个视觉模型 (自动探测属性名)
        vision_enc = None
        for attr in ["visual", "vision_tower", "vision_model"]:
            if hasattr(self.model, attr):
                vision_enc = getattr(self.model, attr)
                break
        if vision_enc is None and hasattr(self.model, "model"):
            for attr in ["visual", "vision_tower", "vision_model"]:
                if hasattr(self.model.model, attr):
                    vision_enc = getattr(self.model.model, attr)
                    break
        if vision_enc is not None:
            for param in vision_enc.parameters():
                param.requires_grad = False
            vision_enc.eval()
            self._vision_enc = vision_enc
        else:
            raise AttributeError("无法找到 Vision Encoder，请检查 transformers 版本")

        # Qwen2-VL hidden_size 在 text_config 下
        tc = getattr(self.model.config, "text_config", self.model.config)
        self.qwen_dim = getattr(tc, "hidden_size", 1536)

    @torch.no_grad()
    def extract(self, image_path: str) -> torch.Tensor:
        """
        提取 V_raw [N, 1536]。

        Args:
            image_path: 图片路径

        Returns:
            V_raw: [N, 1536], float16, on device
        """
        # 构造 messages 以通过 processor
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_path}],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 通过 Qwen 的 visual.get_image_features 获取 V_raw
        # 这个方法内部走 vision_encoder + merger
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # ===== 提取 V_raw: vision_encoder + merger → [N, 1536] =====
        # 方法: vision_enc 取 last_hidden_state → merger 映射
        
        vision_output = self._vision_enc(pixel_values, grid_thw=image_grid_thw)
        raw_features = vision_output.last_hidden_state  # [N, 1280] (ViT embed_dim)

        # merger: Qwen2-VL 的 spatial merge + Linear(1280→1536)
        # merger 通常在 model.visual.merger 或 model.model.visual.merger
        # 接受 (hidden_states, grid_thw) 作为输入 (部分版本)
        vision_module = self._vision_enc
        if hasattr(vision_module, "merger"):
            merger = vision_module.merger
        elif hasattr(vision_module, "deepstack_merger"):
            merger = vision_module.deepstack_merger
        else:
            # 从 visual 模块下面找
            merger = getattr(vision_module, "merger", None)

        if merger is not None:
            image_features = merger(raw_features)
        else:
            image_features = raw_features

        return image_features  # [N, 1536]

    def get_token_count(self, image_path: str) -> int:
        """返回该图像的 visual token 数 N"""
        v_raw = self.extract(image_path)
        return v_raw.shape[0]


class VisualProjector(nn.Module):
    """
    降维投影（已废弃，保留备用）。

    当前架构直接使用 V_raw [N, 1536] 做所有路由计算。
    此类保留供后续可能的 bottleneck 消融实验使用。
    """

    def __init__(self, qwen_dim: int = 1536, structure_dim: int = 1536):
        # 注: VisualProjector 已废弃。当前架构直接使用 V_raw 1536 维做路由计算。
        # 保留此类供后续可能的 bottleneck 实验使用。
        super().__init__()
        self.qwen_dim = qwen_dim
        self.structure_dim = structure_dim

        self.down = nn.Linear(qwen_dim, structure_dim)
        self.up = nn.Linear(structure_dim, qwen_dim)

        # 初始化: 接近恒等映射
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def project_down(self, v_raw: torch.Tensor) -> torch.Tensor:
        """降维投影（已废弃，当前不使用）"""
        return self.down(v_raw)

    def project_up(self, v_struct: torch.Tensor) -> torch.Tensor:
        """升维投影（已废弃，当前不使用）"""
        return self.up(v_struct)

    def forward(self, v_raw: torch.Tensor) -> torch.Tensor:
        """完整降维+升维（已废弃，identity 测试用）"""
        return self.up(self.down(v_raw))

