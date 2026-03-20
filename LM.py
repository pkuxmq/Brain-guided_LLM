from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from nnsight import LanguageModel
import numpy as np
from transformers.generation.utils import GenerationConfig
import torch
from transformers.utils import logging
import os
from accelerate import dispatch_model, infer_auto_device_map
logging.get_logger("transformers").setLevel(logging.ERROR)

class LM_normal():

    def __init__(self, model_path, device="cuda", temperature=0., parse_model_type='default', bf16=False):
        self.device = device
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
        if temperature == 0:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = 1.
            self.model.generation_config.top_p = 1.
            self.model.generation_config.top_k = 50
        else:
            self.model.generation_config.temperature = temperature
        self.model.eval()
        self.tokenizer = tokenizer
        self.parse_model_type = parse_model_type

    def generate_response(self, messages, max_new_tokens=1, get_all_tokens=False):
        eff_max_new_tokens = max_new_tokens
        if self.parse_model_type == 'qwen3':
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        elif self.parse_model_type == 'llama2' and max_new_tokens == 2:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            text += ' '
            encodeds = self.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
            eff_max_new_tokens = 1
        else:
            encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=eff_max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs[0]):]
 
        if get_all_tokens:
            return self.tokenizer.decode(output_ids), generated_ids[0]
        return self.tokenizer.decode(output_ids)
        
    def generate_response_batch(self, messages_all, max_new_tokens=1):
        text_all = []
        eff_max_new_tokens = max_new_tokens
        for messages in messages_all:
            if self.parse_model_type == 'qwen3':
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            elif self.parse_model_type == 'llama2' and max_new_tokens == 2:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                text += ' '
                eff_max_new_tokens = 1
            else:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_all.append(text)
        model_inputs = self.tokenizer(text_all, padding=True, truncation=True, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_new_tokens=eff_max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return response
        
    def __call__(self, messages, max_new_tokens=1, get_all_tokens=False):
        if get_all_tokens:
            ans, all_tokens = self.generate_response(messages, max_new_tokens, get_all_tokens=True)
        else:
            ans = self.generate_response(messages, max_new_tokens)
        if get_all_tokens:
            return ans, all_tokens
        return ans

    def generate_ans(self, messages, max_new_tokens, parse_model_type=None):
        ans = self.generate_response(messages, max_new_tokens)
        return ans

    def generate_ans_batch(self, messages_all, max_new_tokens, parse_model_type=None):
        ans = self.generate_response_batch(messages_all, max_new_tokens)
        return ans


class LM_nnsight():

    def __init__(self, model_path, device="cuda", temperature=0., parse_model_type='default', bf16=False):
        self.device = device
        self.bf16 = bf16
        if bf16:
            if device != 'cpu':
                base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            if device != 'cpu':
                base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = 1.
            base_model.generation_config.top_p = 1.
            base_model.generation_config.top_k = 50
        else:
            base_model.generation_config.temperature = temperature
        base_model.eval()
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
        self.parse_model_type = parse_model_type
        
    def generate_response(self, prompt, max_new_tokens=1, get_all_tokens=False):
        eff_max_new_tokens = max_new_tokens
        if self.parse_model_type == 'qwen3':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        elif self.parse_model_type == 'llama2' and max_new_tokens == 2:
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False)
            text += ' '
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
            eff_max_new_tokens = 1
        else:
            encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(self.device)
        with self.model.generate(max_new_tokens=eff_max_new_tokens) as generator:  
            with generator.invoke(model_inputs) as invoker:
                output = self.model.generator.output.save()
        output_ids = output[0][len(model_inputs[0]):]
        if get_all_tokens:
            return self.model.tokenizer.decode(output_ids), output[0]
        return self.model.tokenizer.decode(output_ids)
                
    def __call__(self, prompt, max_new_tokens=1, get_all_tokens=False):
        if get_all_tokens:
            ans, all_tokens = self.generate_response(prompt, max_new_tokens, get_all_tokens=True)
        else:
            ans = self.generate_response(prompt, max_new_tokens)
        if get_all_tokens:
            return ans, all_tokens
        return ans

    def get_all_states_with_tokens(self, tokens):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []

        model_inputs = tokens.to(self.device)

        with self.model.generate(max_new_tokens=1) as generator:
            with generator.invoke(model_inputs) as invoker:
                for layer in self.model.model.layers:
                    all_attention_states.append(layer.self_attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            if hasattr(HS, 'value'):
                hs_val = HS.value
            else:
                hs_val = HS
            if hasattr(AS, 'value'):
                as_val = AS.value
            else:
                as_val = AS

            # Robust dimensional handling
            if len(hs_val.shape) == 3:
                hs_final = hs_val[0]
            else:
                hs_final = hs_val
                
            if len(as_val.shape) == 3:
                 as_final = as_val[0]
            else:
                 as_final = as_val

            all_hidden_states_numpy.append(hs_final.cpu().float().numpy())
            atts = as_final.cpu().float().numpy()
            
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        # all_hidden_states: (Layers, Tokens, dim)
        # all_attention_states: (Layers, Tokens, Heads, dim_head)
        return all_hidden_states_numpy, all_attention_states_numpy

    def get_all_states(self, prompt, get_input=False):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []

        if self.parse_model_type == 'qwen3':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        elif self.parse_model_type == 'llama2':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False)
            text += ' '
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        else:
            encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(self.device)

        with self.model.generate(max_new_tokens=1) as generator:
            with generator.invoke(model_inputs) as invoker:
                if get_input:
                    embeddings = self.model.model.embed_tokens.output.save()
                for layer in self.model.model.layers:
                    all_attention_states.append(layer.self_attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            if hasattr(HS, 'value'):
                hs_val = HS.value
            else:
                hs_val = HS
            if hasattr(AS, 'value'):
                as_val = AS.value
            else:
                as_val = AS

            # Robust dimensional handling
            if len(hs_val.shape) == 3:
                hs_final = hs_val[0]
            else:
                hs_final = hs_val
                
            if len(as_val.shape) == 3:
                 as_final = as_val[0]
            else:
                 as_final = as_val

            all_hidden_states_numpy.append(hs_final.cpu().float().numpy())
            atts = as_final.cpu().float().numpy()

            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        if get_input:
            if hasattr(embeddings, 'value'):
                emb_val = embeddings.value
            else:
                emb_val = embeddings
            if len(emb_val.shape) == 3:
                embeddings_numpy = emb_val[0].cpu().float().numpy()
            else:
                embeddings_numpy = emb_val.cpu().float().numpy()
            return all_hidden_states_numpy, all_attention_states_numpy, embeddings_numpy
        return all_hidden_states_numpy, all_attention_states_numpy

    def get_all_ans_states_with_tokens(self, tokens, prompt):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []

        model_inputs = tokens.to(self.device)

        with self.model.generate(max_new_tokens=1) as generator:
            with generator.invoke(model_inputs) as invoker:
                for layer in self.model.model.layers:
                    all_attention_states.append(layer.self_attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            if hasattr(HS, 'value'):
                hs_val = HS.value
            else:
                hs_val = HS
            if hasattr(AS, 'value'):
                as_val = AS.value
            else:
                as_val = AS
            # Robust dimensional handling
            if len(hs_val.shape) == 3:
                hs_final = hs_val[0]
            else:
                hs_final = hs_val
                
            if len(as_val.shape) == 3:
                 as_final = as_val[0]
            else:
                 as_final = as_val

            all_hidden_states_numpy.append(hs_final.cpu().float().numpy())
            atts = as_final.cpu().float().numpy()
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        # n_layer * seq_len * dim
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)

        if self.parse_model_type == 'qwen3':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        elif self.parse_model_type == 'llama2': # and max_new_tokens == 2:
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False)
            text += ' '
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
            #max_new_tokens = 1
        else:
            encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        idx = encodeds.shape[1]
        all_hidden_states_numpy = all_hidden_states_numpy[:, idx:]
        all_attention_states_numpy = all_attention_states_numpy[:, idx:]
        
        return all_hidden_states_numpy, all_attention_states_numpy

    def intervention(self, prompt, intervention_layer_index, intervention_rep_diff, max_new_tokens=1):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)

        attention_layer = False
        if intervention_layer_index < n_layers:
            attention_layer = True
        else:
            intervention_layer_index -= n_layers
        intervention_rep_diff = torch.from_numpy(intervention_rep_diff) #.to(self.device)

        if self.parse_model_type == 'qwen3':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        else:
            #encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt")
            encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(self.device)

        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(model_inputs) as invoker:
                # whether new version of nnsight
                tracer = getattr(invoker, "tracer", None)
                use_new_iter = tracer is not None and hasattr(tracer, "iter") and max_new_tokens > 1
                
                if use_new_iter:
                    with tracer.iter[max_new_tokens - 1]:
                        for layer_id, layer in enumerate(self.model.model.layers):
                            if layer_id == intervention_layer_index:
                                if attention_layer:
                                    target_device = layer.self_attn.output[0][0][-1].device
                                    layer.self_attn.output[0][0][-1] += intervention_rep_diff.to(target_device)
                                else:
                                    if hasattr(layer.output, "shape"):
                                        target_device = layer.output[0][-1].device
                                        layer.output[0][-1] += intervention_rep_diff.to(target_device)
                                    else:
                                        target_device = layer.output[0][0][-1].device
                                        layer.output[0][0][-1] += intervention_rep_diff.to(target_device)
                else:
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id == intervention_layer_index:
                            if attention_layer:
                                for idx in range(max_new_tokens-1):
                                    layer.self_attn.next()
                                target_device = layer.self_attn.output[0][0][-1].device
                                layer.self_attn.output[0][0][-1] += intervention_rep_diff.to(target_device)
                            else:
                                for idx in range(max_new_tokens-1):
                                    layer.next()
                                if hasattr(layer.output, "shape"):
                                    target_device = layer.output[0][-1].device
                                    layer.output[0][-1] += intervention_rep_diff.to(target_device)
                                else:
                                    target_device = layer.output[0][0][-1].device
                                    layer.output[0][0][-1] += intervention_rep_diff.to(target_device)
                output = self.model.generator.output.save()

        output_ids = output[0][len(model_inputs[0]):]
        ans = self.model.tokenizer.decode(output_ids)
        return ans

    def intervention_multilayer(self, prompt, intervention_rep_diff_dict, max_new_tokens=1, get_all_intervention_states=False, apply_all_tokens=False):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)

        if get_all_intervention_states:
            all_states = {}

        if self.parse_model_type == 'qwen3':
            text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encodeds = self.model.tokenizer([text], return_tensors='pt')
            encodeds = encodeds.input_ids
        else:
            encodeds = self.model.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to(self.device)

        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(model_inputs) as invoker:
                # whether new version of nnsight
                tracer = getattr(invoker, "tracer", None)
                use_new_iter = tracer is not None and hasattr(tracer, "iter") and max_new_tokens > 1 and not apply_all_tokens
                
                if use_new_iter:
                    with tracer.iter[max_new_tokens - 1]:
                        for layer_id, layer in enumerate(self.model.model.layers):
                            if layer_id in intervention_rep_diff_dict.keys():
                                #layer.self_attn.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(self.device)
                                target_device = layer.self_attn.output[0][0][-1].device
                                layer.self_attn.output[0][0][-1] = layer.self_attn.output[0][0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id] = layer.self_attn.output[0][0][-1].save()
                            if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                                if hasattr(layer.output, "shape"):
                                    #layer.output[0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(self.device)
                                    target_device = layer.output[0][-1].device
                                    layer.output[0][-1] = layer.output[0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                    if get_all_intervention_states:
                                        all_states[layer_id + n_layers] = layer.output[0][-1].save()
                                else:
                                    #layer.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(self.device)
                                    target_device = layer.output[0][0][-1].device
                                    layer.output[0][0][-1] = layer.output[0][0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                    if get_all_intervention_states:
                                        all_states[layer_id + n_layers] = layer.output[0][0][-1].save()
                else:
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.self_attn.next()
                            target_device = layer.self_attn.output[0][0][-1].device
                            layer.self_attn.output[0][0][-1] = layer.self_attn.output[0][0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                            if get_all_intervention_states:
                                all_states[layer_id] = layer.self_attn.output[0][0][-1].save()
                        if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.next()
                            if hasattr(layer.output, "shape"):
                                target_device = layer.output[0][-1].device
                                layer.output[0][-1] = layer.output[0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id + n_layers] = layer.output[0][-1].save()
                            else:
                                target_device = layer.output[0][0][-1].device
                                layer.output[0][0][-1] = layer.output[0][0][-1] + torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id + n_layers] = layer.output[0][0][-1].save()
                output = self.model.generator.output.save()

        output_ids = output[0][len(model_inputs[0]):]
        ans = self.model.tokenizer.decode(output_ids)

        if get_all_intervention_states:
            for layer_id in all_states.keys():
                if hasattr(all_states[layer_id], 'value'):
                    all_states[layer_id] = all_states[layer_id].value.cpu().float().numpy()
                else:
                    all_states[layer_id] = all_states[layer_id].cpu().float().numpy()
            return ans, all_states

        return ans

    def intervention_multilayer_batch(self, messages_all, intervention_rep_diff_dict, max_new_tokens=1, get_all_intervention_states=False, apply_all_tokens=False):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)

        if get_all_intervention_states:
            all_states = {}

        text_all = []
        for prompt in messages_all:
            if self.parse_model_type == 'qwen3':
                text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                text = self.model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            text_all.append(text)
        encodeds = self.model.tokenizer(text_all, padding=True, truncation=True, return_tensors="pt")
        model_inputs = encodeds.to(self.device)

        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(model_inputs) as invoker:
                # whether new version of nnsight
                tracer = getattr(invoker, "tracer", None)
                use_new_iter = tracer is not None and hasattr(tracer, "iter") and max_new_tokens > 1 and not apply_all_tokens
                
                if use_new_iter:
                    with tracer.iter[max_new_tokens - 1]:
                        for layer_id, layer in enumerate(self.model.model.layers):
                            if layer_id in intervention_rep_diff_dict.keys():
                                target_device = layer.self_attn.output[0][:, -1].device
                                layer.self_attn.output[0][:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id] = layer.self_attn.output[0][:, -1].save()
                            if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                                if hasattr(layer.output, "shape"):
                                    target_device = layer.output[:, -1].device
                                    layer.output[:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                    if get_all_intervention_states:
                                        all_states[layer_id + n_layers] = layer.output[:, -1].save()
                                else:
                                    target_device = layer.output[0][:, -1].device
                                    layer.output[0][:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                    if get_all_intervention_states:
                                        all_states[layer_id + n_layers] = layer.output[0][:, -1].save()
                else:
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.self_attn.next()
                            target_device = layer.self_attn.output[0][:, -1].device
                            layer.self_attn.output[0][:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                            if get_all_intervention_states:
                                all_states[layer_id] = layer.self_attn.output[0][:, -1].save()
                        if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.next()
                            if hasattr(layer.output, "shape"):
                                target_device = layer.output[:, -1].device
                                layer.output[:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id + n_layers] = layer.output[:, -1].save()
                            else:
                                target_device = layer.output[0][:, -1].device
                                layer.output[0][:, -1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                if get_all_intervention_states:
                                    all_states[layer_id + n_layers] = layer.output[0][:, -1].save()
                output = self.model.generator.output.save()

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)]
        ans = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if get_all_intervention_states:
            for layer_id in all_states.keys():
                if hasattr(all_states[layer_id], 'value'):
                    all_states[layer_id] = all_states[layer_id].value.cpu().float().numpy()
                else:
                    all_states[layer_id] = all_states[layer_id].cpu().float().numpy()
            return ans, all_states

        return ans


class LM_untrained_nnsight(LM_nnsight):

    def __init__(self, model_path, device="cuda", temperature=0., parse_model_type='default', bf16=False):
        self.device = device
        self.bf16 = bf16
        #config_path = os.path.join(model_path, 'config.json')
        #config = AutoConfig.from_pretrained(config_path)
        config = AutoConfig.from_pretrained(model_path)
        multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        # random initialization
        if bf16:
            if device == 'cpu' or not multi_gpu:
                base_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).to(device)
            else:
                base_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
                no_split_modules = base_model._no_split_modules if hasattr(base_model, "_no_split_modules") else None
                max_mem = int(0.9 * torch.cuda.get_device_properties(0).total_memory)
                max_memory = {i: max_mem for i in range(torch.cuda.device_count())}
                device_map = infer_auto_device_map(base_model, max_memory=max_memory, no_split_module_classes=no_split_modules)
                base_model = dispatch_model(base_model, device_map=device_map)
        else:
            if device == 'cpu' or not multi_gpu:
                base_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).to(device)
            else:
                base_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
                no_split_modules = base_model._no_split_modules if hasattr(base_model, "_no_split_modules") else None
                max_mem = int(0.9 * torch.cuda.get_device_properties(0).total_memory)
                max_memory = {i: max_mem for i in range(torch.cuda.device_count())}
                device_map = infer_auto_device_map(base_model, max_memory=max_memory, no_split_module_classes=no_split_modules)
                base_model = dispatch_model(base_model, device_map=device_map)
                
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = 1.
            base_model.generation_config.top_p = 1.
            base_model.generation_config.top_k = 50
        else:
            base_model.generation_config.temperature = temperature
        base_model.eval()
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
        self.parse_model_type = parse_model_type


class LM_nnsight_base():

    def __init__(self, model_path, device="cuda", temperature=0.):
        self.device = device
        if device != 'cpu':
            base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = 1.
            base_model.generation_config.top_p = 1.
            base_model.generation_config.top_k = 50
        else:
            base_model.generation_config.temperature = temperature
        base_model.eval()
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
        
    def generate_response(self, prompt, max_new_tokens=1, get_all_tokens=False):
        model_inputs = prompt
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
            with generator.invoke(model_inputs) as invoker:
                output = self.model.generator.output.save()
        if get_all_tokens:
            return self.model.tokenizer.decode(output[0][-1]), output[0]
        return self.model.tokenizer.decode(output[0][-1])
        
    def __call__(self, prompt, max_new_tokens=1, get_all_tokens=False):
        if get_all_tokens:
            ans, all_tokens = self.generate_response(prompt, max_new_tokens, get_all_tokens=True)
        else:
            ans = self.generate_response(prompt, max_new_tokens)
        if get_all_tokens:
            return ans, all_tokens
        return ans

    def get_all_states_with_tokens(self, tokens):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []

        model_inputs = tokens.to(self.device)

        with self.model.generate(max_new_tokens=1) as generator:
            with generator.invoke(model_inputs) as invoker:
                for layer in self.model.model.layers:
                    all_attention_states.append(layer.self_attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            if hasattr(HS, 'value'):
                hs_val = HS.value
            else:
                hs_val = HS
            if hasattr(AS, 'value'):
                as_val = AS.value
            else:
                as_val = AS

            # Robust dimensional handling
            if len(hs_val.shape) == 3:
                hs_final = hs_val[0]
            else:
                hs_final = hs_val
                
            if len(as_val.shape) == 3:
                 as_final = as_val[0]
            else:
                 as_final = as_val

            all_hidden_states_numpy.append(hs_final.cpu().float().numpy())
            atts = as_final.cpu().float().numpy()
            
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        # all_hidden_states: (Layers, Tokens, dim)
        # all_attention_states: (Layers, Tokens, Heads, dim_head)
        return all_hidden_states_numpy, all_attention_states_numpy

    def get_all_states(self, prompt, get_input=False):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []

        encodeds = self.model.tokenizer.encode(prompt)
        model_inputs = encodeds.to(self.device)

        with self.model.generate(max_new_tokens=1) as generator:
            with generator.invoke(model_inputs) as invoker:
                if get_input:
                    embeddings = self.model.model.embed_tokens.output.save()
                for layer in self.model.model.layers:
                    all_attention_states.append(layer.self_attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            if hasattr(HS, 'value'):
                hs_val = HS.value
            else:
                hs_val = HS
            if hasattr(AS, 'value'):
                as_val = AS.value
            else:
                as_val = AS

            # Robust dimensional handling
            if len(hs_val.shape) == 3:
                hs_final = hs_val[0]
            else:
                hs_final = hs_val
                
            if len(as_val.shape) == 3:
                 as_final = as_val[0]
            else:
                 as_final = as_val

            all_hidden_states_numpy.append(hs_final.cpu().float().numpy())
            atts = as_final.cpu().float().numpy()
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        if get_input:
            if hasattr(embeddings, 'value'):
                emb_val = embeddings.value
            else:
                emb_val = embeddings
            if len(emb_val.shape) == 3:
                embeddings_numpy = emb_val[0].cpu().float().numpy()
            else:
                embeddings_numpy = emb_val.cpu().float().numpy()
            return all_hidden_states_numpy, all_attention_states_numpy, embeddings_numpy
        return all_hidden_states_numpy, all_attention_states_numpy

    def intervention(self, prompt, intervention_layer_index, intervention_rep_diff, max_new_tokens=1):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)

        attention_layer = False
        if intervention_layer_index < n_layers:
            attention_layer = True
        else:
            intervention_layer_index -= n_layers
        intervention_rep_diff = torch.from_numpy(intervention_rep_diff) #.to(self.device)

        model_inputs = prompt

        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(model_inputs) as invoker:
                # whether new version of nnsight
                tracer = getattr(invoker, "tracer", None)
                use_new_iter = tracer is not None and hasattr(tracer, "iter") and max_new_tokens > 1
                
                if use_new_iter:
                    with tracer.iter[max_new_tokens - 1]:
                        for layer_id, layer in enumerate(self.model.model.layers):
                            if layer_id == intervention_layer_index:
                                if attention_layer:
                                    target_device = layer.self_attn.output[0][0][-1].device
                                    layer.self_attn.output[0][0][-1] += intervention_rep_diff.to(target_device)
                                else:
                                    if hasattr(layer.output, "shape"):
                                        target_device = layer.output[0][-1].device
                                        layer.output[0][-1] += intervention_rep_diff.to(target_device)
                                    else:
                                        target_device = layer.output[0][0][-1].device
                                        layer.output[0][0][-1] += intervention_rep_diff.to(target_device)
                else:
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id == intervention_layer_index:
                            if attention_layer:
                                for idx in range(max_new_tokens-1):
                                    layer.self_attn.next()
                                target_device = layer.self_attn.output[0][0][-1].device
                                layer.self_attn.output[0][0][-1] += intervention_rep_diff.to(target_device)
                            else:
                                for idx in range(max_new_tokens-1):
                                    layer.next()
                                if hasattr(layer.output, "shape"):
                                    target_device = layer.output[0][-1].device
                                    layer.output[0][-1] += intervention_rep_diff.to(target_device)
                                else:
                                    target_device = layer.output[0][0][-1].device
                                    layer.output[0][0][-1] += intervention_rep_diff.to(target_device)
                output = self.model.generator.output.save()

        ans = self.model.tokenizer.decode(output[0][-max_new_tokens:])
        return ans

    def intervention_multilayer(self, prompt, intervention_rep_diff_dict, max_new_tokens=1, apply_all_tokens=False):
        # now only for attention
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)

        model_inputs = prompt

        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(model_inputs) as invoker:
                # whether new version of nnsight
                tracer = getattr(invoker, "tracer", None)
                use_new_iter = tracer is not None and hasattr(tracer, "iter") and max_new_tokens > 1 and not apply_all_tokens
                
                if use_new_iter:
                    with tracer.iter[max_new_tokens - 1]:
                        for layer_id, layer in enumerate(self.model.model.layers):
                            if layer_id in intervention_rep_diff_dict.keys():
                                target_device = layer.self_attn.output[0][0][-1].device
                                layer.self_attn.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                            if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                                if hasattr(layer.output, "shape"):
                                    target_device = layer.output[0][-1].device
                                    layer.output[0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                                else:
                                    target_device = layer.output[0][0][-1].device
                                    layer.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                else:
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.self_attn.next()
                            target_device = layer.self_attn.output[0][0][-1].device
                            layer.self_attn.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id]).to(target_device)
                        if (layer_id + n_layers) in intervention_rep_diff_dict.keys():
                            if not apply_all_tokens:
                                for idx in range(max_new_tokens-1):
                                    layer.next()
                            if hasattr(layer.output, "shape"):
                                target_device = layer.output[0][-1].device
                                layer.output[0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                            else:
                                target_device = layer.output[0][0][-1].device
                                layer.output[0][0][-1] += torch.from_numpy(intervention_rep_diff_dict[layer_id+n_layers]).to(target_device)
                output = self.model.generator.output.save()

        ans = self.model.tokenizer.decode(output[0][-max_new_tokens:])
        return ans
