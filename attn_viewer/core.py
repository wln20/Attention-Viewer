import os
import torch
from .plot_utils import plot_heatmap

# a wrapper
def view_attention(
    model=None,  # the model object
    model_id=None,
    tokenizer=None,
    prompt=None,
    save_attention_scores=False,
    save_attention_scores_path=None,
    load_attention_scores_path=None,
    save_fig_path=None,
    num_figs_per_row=4
):
    if load_attention_scores_path:  # plot using the existing attention scores
        with open(load_attention_scores_path, 'rb') as f:
            saved_data = torch.load(f)
            attention_scores = saved_data['attention_scores']
            tokens_list = saved_data['tokens_list']

    else:
        assert model is not None and model_id is not None and prompt is not None and tokenizer is not None, \
            "`model`, `model_id`, `tokenizer` and `prompt` must all be specified without `load_attention_scores_path`!"
            
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
        tokens_list = list(map(lambda x:x.replace('▁',''), tokenizer.convert_ids_to_tokens(inputs[0].cpu())))   # used as labels when plotting
        print("* Generating ...")
        with torch.no_grad():
            attention_scores = model(inputs, output_attentions=True)['attentions'] # a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]
        attention_scores = [attention_scores_layer.detach().cpu() for attention_scores_layer in attention_scores]
        
        if save_attention_scores:  # each layer's attention scores is stored in one safetensors file
            assert save_attention_scores_path is not None, \
                "`save_attention_scores_path` must be specified to save attention scores!"
            print('* Saving attention scores ...')
            save_path = save_attention_scores_path
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f'{model_id}_attn_scores.pt'), 'wb') as f:
                saved_data = {
                    'attention_scores': attention_scores,
                    'tokens_list': tokens_list
                }
                torch.save(saved_data, f)

    print('Plotting heatmap for attention scores ...')
    plot_heatmap(attention_scores, model_id, save_fig_path, tokens_list)
