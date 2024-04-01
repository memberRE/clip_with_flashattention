from types import MethodType
from typing import Optional

from transformers import AutoProcessor, AutoModel, AutoConfig, CLIPModel
from PIL import Image
import requests
import torch.nn as nn
import torch
from transformers.models.clip.modeling_clip import CLIPAttention
from flash_attn import flash_attn_func

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



    # get query proj
    # query_states = self.q_proj(hidden_states) * self.scale
    # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    # key_states = key_states.view(*proj_shape)
    # value_states = value_states.view(*proj_shape)
    #
    # src_len = key_states.size(1)

    # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    #
    # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )
    #
    # # apply the causal_attention_mask first
    # if causal_attention_mask is not None:
    #     if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
    #             f" {causal_attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    #
    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    #
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    #
    # if output_attentions:
    #     # this operation is a bit akward, but it's required to
    #     # make sure that attn_weights keeps its gradient.
    #     # In order to do so, attn_weights have to reshaped
    #     # twice and have to be reused in the following
    #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    # else:
    #     attn_weights_reshaped = None
    #
    # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    #
    # attn_output = torch.bmm(attn_probs, value_states)
    #
    # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    # attn_output = attn_output.transpose(1, 2)
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

def get_test_input():
    processor = AutoProcessor.from_pretrained("/home/JJ_Group/cheny/clip-vit-large-patch14-336")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    return inputs

if __name__ == '__main__':
    model = get_orgin_model()
    apply_flash_attention(model)
    inputs = get_test_input().to('cuda:0')
    outputs = model(**inputs)
    print(outputs)