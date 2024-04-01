import random
from types import MethodType
from typing import Optional

from transformers import AutoProcessor, AutoModel, AutoConfig, CLIPModel
from PIL import Image, ImageFilter
import requests
import torch.nn as nn
import torch
from transformers.models.clip.modeling_clip import CLIPAttention
from flash_attn import flash_attn_func
import time

# class CLIPModel_Flash(CLIPModel):
#     pass

def flash_attention_forward(self,
                            hidden_states: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            causal_attention_mask: Optional[torch.Tensor] = None,
                            output_attentions: Optional[bool] = False,):
    """Input shape: Batch x Time x Channel"""

    def _shape_flash_attention(tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    bsz, tgt_len, embed_dim = hidden_states.size()

    # b, seq_len, num_heads, head_dim
    query_states = _shape_flash_attention(self.q_proj(hidden_states), -1, bsz)
    key_states = _shape_flash_attention(self.k_proj(hidden_states), -1, bsz)
    value_states = _shape_flash_attention(self.v_proj(hidden_states), -1, bsz)
    # batch_size, seqlen, nheads, headdim
    attn_output = flash_attn_func(query_states, key_states, value_states, causal=causal_attention_mask is not None)

    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None

def apply_flash_attention(model: nn.Module):
    for module in model.modules():
        if isinstance(module, CLIPAttention):
            module.forward = MethodType(flash_attention_forward, module)
def get_orgin_model(path="/home/JJ_Group/cheny/clip-vit-large-patch14-336"):
    config = AutoConfig.from_pretrained("/home/JJ_Group/cheny/clip-vit-large-patch14-336")
    # model = AutoModel.from_config(config)
    model = CLIPModel.from_pretrained(path,
                                      config=config,
                                      torch_dtype=torch.float16,
                                      device_map='cuda:0'
                                      )
    return model

def get_test_input(text_bs=1, image_bs=1):
    import copy
    processor = AutoProcessor.from_pretrained("/home/JJ_Group/cheny/clip-vit-large-patch14-336")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    images = [copy.deepcopy(image) for _ in range(image_bs)]
    for img in images:
        noisy_image = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))
        img.paste(noisy_image)
    text = ["a photo of a cat", "a photo of a dog"]
    for i in range(text_bs - 1):
        text.append("a photo of a cat")
        text.append("a photo of a dog")
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    return inputs

if __name__ == '__main__':
    model = get_orgin_model()
    model2 = get_orgin_model()
    apply_flash_attention(model2)
    inputs = get_test_input(64, 64).to('cuda:0')
    with torch.no_grad():
        model2(**inputs)
        model(**inputs)
    times_ori = []
    times_flash = []
    with torch.no_grad():
        for i in range(4):
            inputs = get_test_input(64, 64).to('cuda:0')
            start_time = time.time()
            outputs2 = model2(**inputs)
            end_time = time.time()
            times_flash.append(end_time-start_time)
        for i in range(4):
            inputs = get_test_input(64, 64).to('cuda:0')
            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()
            times_ori.append(end_time-start_time)
    mean_ori = sum(times_ori) / len(times_ori)
    mean_flash = sum(times_flash) / len(times_flash)
    print(mean_ori)
    print(mean_flash)
    print('加速比: ', mean_ori / mean_flash)
