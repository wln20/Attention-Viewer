import os
import math
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list=None, num_figs_per_row=4):
    """
    attention_scores: a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]
    tokens_list: act as xticks and yticks of the figure, eg. ['<s>', 'Hi', ',', 'how', 'are', 'you', '?']
    """
    save_fig_path_model = os.path.join(save_fig_path, model_id) # the model's results are saved under this dir 
    os.makedirs(save_fig_path_model, exist_ok=True)

    # a figure for all
    print(f'plotting a figure for all layers ...')
    num_heads = len(attention_scores)
    num_rows = math.ceil(num_heads / num_figs_per_row) 
    fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
    for layer_idx in tqdm(range(len(attention_scores))):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        avg_attention_scores = attention_scores[layer_idx][0].mean(dim=0)    # [ seq_len, seq_len]
        mask = torch.triu(torch.ones_like(avg_attention_scores, dtype=torch.bool), diagonal=1)
        sns.heatmap(avg_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
        axes[row, col].set_title(f'layer {layer_idx}')

    plt.suptitle(f'all layers avg') 
    plt.savefig(os.path.join(save_fig_path_model, f'all_layers_avg.jpg'))
    plt.close()   

    if not plot_figs_per_head:
        return

    # a figure for each layer
    for layer_idx in range(len(attention_scores)):
        print(f'plotting layer {layer_idx} ...')
        num_heads = attention_scores[layer_idx].shape[1]
        num_rows = math.ceil(num_heads / num_figs_per_row)
        fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
        for head_idx in tqdm(range(num_heads)):
            row, col = head_idx // num_figs_per_row, head_idx % num_figs_per_row
            head_attention_scores = attention_scores[layer_idx][0][head_idx]    # [seq_len, seq_len]
            mask = torch.triu(torch.ones_like(head_attention_scores, dtype=torch.bool), diagonal=1)
            sns.heatmap(head_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
            axes[row, col].set_title(f'head {head_idx}')

        plt.suptitle(f'layer_{layer_idx}') 
        plt.savefig(os.path.join(save_fig_path_model, f'layer_{layer_idx}.jpg'))
        plt.close()
