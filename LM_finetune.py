import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import LoraConfig, get_peft_model


class ModelwithAttentionSupervision(nn.Module):
    def __init__(self, model_path, layers_to_supervise, device='cuda', local=False, lora=False, bf16=False, model_type='default'):
        super(ModelwithAttentionSupervision, self).__init__()
        self.device = device
        self.model_type = model_type

        self.bf16 = bf16
        if bf16:
            if device != 'cpu':
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            if device != 'cpu':
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.
        self.model.generation_config.top_p = 1.
        self.model.generation_config.top_k = 50

        self.tokenizer = tokenizer

        self.layers_to_supervise = layers_to_supervise
        self.max_layer_num = max(layers_to_supervise)

        self.attention_states = {}
        self.local = local
        self.lora = lora
        if lora:
            # apply LoRA
            lora_config = LoraConfig(
                r=8,  # Rank of LoRA
                lora_alpha=16,  # Scaling factor
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention heads
                lora_dropout=0.0  # Dropout for LoRA (default: 0)
            )
            for i, layer in enumerate(self.model.model.layers):
                if i in layers_to_supervise:
                    layer.self_attn = get_peft_model(layer.self_attn, lora_config)
            # freeze other parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.model.model.layers):
                if i in layers_to_supervise:
                    for name, param in layer.self_attn.named_parameters():
                        if 'lora' in name:
                            param.requires_grad = True
        else:
            # freeze other parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.model.model.layers):
                if i in layers_to_supervise:
                    for name, param in layer.self_attn.named_parameters():
                        param.requires_grad = True

        self.hook_handles = []
        self.apply_attention_hook()

        self.mse_loss_fn = nn.MSELoss()

    def apply_attention_hook(self):
        for layer_index in self.layers_to_supervise:
            layer = self.model.model.layers[layer_index]
            layer.self_attn.register_forward_hook(self.save_attention_state_hook(layer_index))

    # for local loss
    def apply_detach_hook(self):
        for layer_index in self.layers_to_supervise:
            if layer_index > 0:
                layer = self.model.model.layers[layer_index-1]
                handle = layer.register_forward_hook(self.detach_output_hook())
                self.hook_handles.append(handle)

    def remove_detach_hook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def merge_lora(self):
        if self.lora:
            for i, layer in enumerate(self.model.model.layers):
                if i in self.layers_to_supervise:
                    layer.self_attn = layer.self_attn.merge_and_unload()

    def save_attention_state_hook(self, layer_index):
        def hook(module, input, output):
            # last token
            self.attention_states[layer_index] = output[0][0, -1]
        return hook

    def detach_output_hook(self):
        def hook(module, input, output):
            output_ = (output[0].detach(), *output[1:])
            return output_
        return hook

    def forward(self, input_ids, attention_mask=None, attention_supervision_dict=None, label=None, new_direction_scale=1.):
        if label is None:
            # apply our representational supervision
            assert attention_supervision_dict is not None
            self.attention_states = {}
            if self.local and len(self.hook_handles) == 0:
                self.apply_detach_hook()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = None
            for layer_index in self.layers_to_supervise:
                attention_state = self.attention_states[layer_index]
                sup_info = attention_supervision_dict[layer_index]
                if isinstance(sup_info, dict):
                    # fmri supervision
                    W = sup_info['W'].to(attention_state.dtype).to(attention_state.device)
                    fmri_state = sup_info['fmri_state'].to(attention_state.dtype).to(attention_state.device)
                    pred_state = torch.matmul(attention_state, W.T)
                    if 'b' in sup_info.keys():
                        b = sup_info['b'].to(attention_state.dtype).to(attention_state.device)
                        pred_state = pred_state + b
                    if sup_info['loss_type'] == 'mse':
                        loss_ = self.mse_loss_fn(pred_state, fmri_state)
                    else:
                        pred_state = pred_state / (torch.norm(pred_state) + 1e-6)
                        fmri_state = fmri_state / (torch.norm(fmri_state) + 1e-6)
                        cos_sim = torch.sum(pred_state * fmri_state)
                        loss_ = 1. - torch.mean(cos_sim)
                    if loss is None:
                        loss = loss_
                    else:
                        loss += loss_.to(loss.device)
                elif isinstance(sup_info, list) and isinstance(sup_info[0], dict):
                    # multiple fmri supervision
                    num = len(sup_info)
                    for sup_info_ in sup_info:
                        # fmri supervision
                        W = sup_info_['W'].to(attention_state.dtype).to(attention_state.device)
                        fmri_state = sup_info_['fmri_state'].to(attention_state.dtype).to(attention_state.device)
                        pred_state = torch.matmul(attention_state, W.T)
                        if 'b' in sup_info_.keys():
                            b = sup_info_['b'].to(attention_state.dtype).to(attention_state.device)
                            pred_state = pred_state + b
                        if sup_info_['loss_type'] == 'mse':
                            loss_ = self.mse_loss_fn(pred_state, fmri_state)
                        else:
                            pred_state = pred_state / (torch.norm(pred_state) + 1e-6)
                            fmri_state = fmri_state / (torch.norm(fmri_state) + 1e-6)
                            cos_sim = torch.sum(pred_state * fmri_state)
                            loss_ = (1. - torch.mean(cos_sim)) / num
                        if loss is None:
                            loss = loss_
                        else:
                            loss += loss_.to(loss.device)
                elif isinstance(sup_info, list):
                    # multiple intervened point
                    num = len(sup_info)
                    for i, sup in enumerate(sup_info):
                        attention_supervision = torch.from_numpy(sup).to(attention_state.dtype).to(attention_state.device)
                        loss_ = self.mse_loss_fn(attention_state, attention_supervision) / num
                        if i > 0:
                            loss_ = loss_ * new_direction_scale
                        if loss is None:
                            loss = loss_
                        else:
                            loss += loss_.to(loss.device)
                else:
                    # intervened point supervision, or regularization
                    attention_supervision = torch.from_numpy(sup_info).to(attention_state.dtype).to(attention_state.device)
                    loss_ = self.mse_loss_fn(attention_state, attention_supervision)
                    if loss is None:
                        loss = loss_
                    else:
                        loss += loss_.to(loss.device)
            return loss
        else:
            # output language supervision
            if len(self.hook_handles) > 0:
                self.remove_detach_hook()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits[:,-1]
            loss_label = loss_fct(logits, label)
            return loss_label

    # TODO: not consider transformers update
    def forward_partial(self, input_ids, attention_mask=None, attention_supervision_dict=None, label=None, new_direction_scale=1.):
        assert label is None
        assert attention_supervision_dict is not None
        self.attention_states = {}
        if self.local and len(self.hook_handles) == 0:
            self.apply_detach_hook()
        # only inference until the last supervision layer
        # ref: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
        # ref: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/models/mistral/modeling_mistral.py
        # ref: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/models/llama/modeling_llama.py
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        cache_position = torch.arange(0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        if self.model_type == 'mistral':
            causal_mask = self.model.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False, False)
        else:
            causal_mask = self.model.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
        hidden_states = inputs_embeds
        max_layer_num = max(self.layers_to_supervise)
        for i, decoder_layer in enumerate(self.model.model.layers):
            if i > max_layer_num:
                break
            layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, cache_position=cache_position)
            hidden_states = layer_outputs[0]
        # inference done
        loss = None
        for layer_index in self.layers_to_supervise:
            attention_state = self.attention_states[layer_index]
            sup_info = attention_supervision_dict[layer_index]
            if isinstance(sup_info, dict):
                # fmri supervision
                W = sup_info['W'].to(attention_state.dtype).to(attention_state.device)
                fmri_state = sup_info['fmri_state'].to(attention_state.dtype).to(attention_state.device)
                pred_state = torch.matmul(attention_state, W.T)
                if 'b' in sup_info.keys():
                    b = sup_info['b'].to(attention_state.dtype).to(attention_state.device)
                    pred_state = pred_state + b
                if sup_info['loss_type'] == 'mse':
                    loss_ = self.mse_loss_fn(pred_state, fmri_state)
                else:
                    pred_state = pred_state / (torch.norm(pred_state) + 1e-6)
                    fmri_state = fmri_state / (torch.norm(fmri_state) + 1e-6)
                    cos_sim = torch.sum(pred_state * fmri_state)
                    loss_ = 1. - torch.mean(cos_sim)
                if loss is None:
                    loss = loss_
                else:
                    loss += loss_.to(loss.device)
            elif isinstance(sup_info, list) and isinstance(sup_info[0], dict):
                # multiple fmri supervision
                num = len(sup_info)
                for sup_info_ in sup_info:
                    # fmri supervision
                    W = sup_info_['W'].to(attention_state.dtype).to(attention_state.device)
                    fmri_state = sup_info_['fmri_state'].to(attention_state.dtype).to(attention_state.device)
                    pred_state = torch.matmul(attention_state, W.T)
                    if 'b' in sup_info_.keys():
                        b = sup_info_['b'].to(attention_state.dtype).to(attention_state.device)
                        pred_state = pred_state + b
                    if sup_info_['loss_type'] == 'mse':
                        loss_ = self.mse_loss_fn(pred_state, fmri_state)
                    else:
                        pred_state = pred_state / (torch.norm(pred_state) + 1e-6)
                        fmri_state = fmri_state / (torch.norm(fmri_state) + 1e-6)
                        cos_sim = torch.sum(pred_state * fmri_state)
                        loss_ = (1. - torch.mean(cos_sim)) / num
                    if loss is None:
                        loss = loss_
                    else:
                        loss += loss_.to(loss.device)
            elif isinstance(sup_info, list):
                # multiple intervened point
                num = len(sup_info)
                for i, sup in enumerate(sup_info):
                    attention_supervision = torch.from_numpy(sup).to(attention_state.dtype).to(attention_state.device)
                    loss_ = self.mse_loss_fn(attention_state, attention_supervision) / num
                    if i > 0:
                        loss_ = loss_ * new_direction_scale
                    if loss is None:
                        loss = loss_
                    else:
                        loss += loss_.to(loss.device)
            else:
                attention_supervision = torch.from_numpy(sup_info).to(attention_state.dtype).to(attention_state.device)
                loss_ = self.mse_loss_fn(attention_state, attention_supervision)
                if loss is None:
                    loss = loss_
                else:
                    loss += loss_.to(loss.device)
        return loss

    def generate_ans(self, messages, max_new_tokens, parse_model_type='default'):
        device = self.device

        eff_max_new_tokens = max_new_tokens
        if parse_model_type == 'qwen3':
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        elif parse_model_type == 'llama2' and max_new_tokens == 2:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            text += ' '
            encodeds = self.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
            eff_max_new_tokens = 1
        else:
            encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=eff_max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs[0]):]
        ans = self.tokenizer.decode(output_ids)
        return ans

    def generate_ans_batch(self, messages_all, max_new_tokens, parse_model_type='qwen'):
        device = self.device

        text_all = []
        eff_max_new_tokens = max_new_tokens
        for messages in messages_all:
            if parse_model_type == 'qwen3':
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            elif parse_model_type == 'llama2' and max_new_tokens == 2:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                text += ' '
                eff_max_new_tokens = 1
            else:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_all.append(text)
        model_inputs = self.tokenizer(text_all, padding=True, truncation=True, return_tensors="pt").to(device)
        generated_ids = self.model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_new_tokens=eff_max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = [r.strip() for r in response]
        return response

    def get_attention_state(self, input_ids, attention_mask=None):
        self.attention_states = {}
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.attention_states

