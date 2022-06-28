
<img src="./SMART.png"></img>


# SMART - PyTorch

A PyTorch implementation of <a href="https://aclanthology.org/2020.acl-main.197.pdf">SMART</a>, a regularization technique to fine-tune pretrained (language) models. You might also be interested in <a href="https://github.com/archinetai/vat-pytorch">vat-pytorch</a>, a more generic collection of virtual adversarial training (VAT) methods, in PyTorch. 

## Install

```bash
$ pip install smart-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/smart-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/smart-pytorch/) 

## Usage

### Minimal Example

```py
import torch
import torch.nn as nn
from smart_pytorch import SMARTLoss

# Define function that will be perturbed (usually our network)
eval_fn = torch.nn.Linear(in_features=10, out_features=20)

# Define loss function between states 
loss_fn = nn.MSELoss()

# Initialize regularization loss
regularizer = SMARTLoss(eval_fn = eval_fn, loss_fn = loss_fn)

# Compute initial input embed and output state 
embed = torch.rand(1, 10) # [batch_size, in_features]
state = eval_fn(embed) # [batch_size, out_featueres]

# Compute regularation loss 
loss = regularizer(embed, state)
loss # tensor(0.0922578126, grad_fn=<MseLossBackward0>)
```

Where `eval_fn` is a function (usually a neural network) that takes as input an embedding `embed` and produces as output one or multiple states `state`. Internally, this function is used to perturb the input `embed` with noise to get a perturbed `state` which is compared with the initially provided `state`. 

### Full API Example 
```python
import torch
import torch.nn as nn
from smart_pytorch import SMARTLoss

# Define function that will be perturbed (usually our network)
eval_fn = torch.nn.Linear(in_features=10, out_features=20)

# Define loss function between states 
loss_fn = nn.MSELoss()

# Norm used to normalize the gradient 
inf_norm = lambda x: torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

# Initialize regularization loss
regularizer = SMARTLoss(
    eval_fn = eval_fn,      
    loss_fn = loss_fn,      # Loss to apply between perturbed and true state 
    loss_last_fn = loss_fn, # Loss to apply between perturbed and true state on the last iteration (default = loss_fn)
    norm_fn = inf_norm,     # Norm used to normalize the gradient (default = inf_norm)
    num_steps = 1,          # Number of optimization steps to find noise (default = 1)
    step_size = 1e-3,       # Step size to improve noise (default = 1e-3)
    epsilon = 1e-6,         # Noise norm constraint (default = 1e-6)
    noise_var = 1e-5        # Initial noise variance (default = 1e-5)
)

# Compute initial input embed and output state 
embed = torch.rand(1, 10) # [batch_size, in_features]
state = eval_fn(embed) # [batch_size, out_featueres]

# Compute regularation loss 
loss = regularizer(embed, state)
loss # tensor(0.0432184562, grad_fn=<MseLossBackward0>)
```

### RoBERTa Classification Example

This example demostrates how to wrap a RoBERTa classifier from Huggingface to use with SMART.

```py
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SMARTRobertaClassificationModel(nn.Module):
    
    def __init__(self, model, weight = 0.02):
        super().__init__()
        self.model = model 
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):

        # Get initial embeddings 
        embed = self.model.roberta.embeddings(input_ids) 

        # Define eval function 
        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask)
            pooled = outputs[0] 
            logits = self.model.classifier(pooled) 
            return logits 
        
        # Define SMART loss
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        # Compute initial (unperturbed) state 
        state = eval(embed)
        # Apply classification loss 
        loss = F.cross_entropy(state.view(-1, 2), labels.view(-1))
        # Apply smart loss 
        loss += self.weight * smart_loss_fn(embed, state)
        
        return state, loss
    

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base')  

model_smart = SMARTRobertaClassificationModel(model)
# Compute inputs 
text = ["This text belongs to class 1...", "This text belongs to class 0..."]
inputs = tokenizer(text, return_tensors='pt')
labels = torch.tensor([1, 0]) 

# Compute output and loss 
state, loss = model_smart(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], labels = labels)
print(state.shape, loss) # torch.Size([2, 2]) tensor(0.6980957389, grad_fn=<AddBackward0>)
```




## Citations

```bibtex
@inproceedings{Jiang2020SMARTRA,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Haoming Jiang and Pengcheng He and Weizhu Chen and Xiaodong Liu and Jianfeng Gao and Tuo Zhao},
  booktitle={ACL},
  year={2020}
}
```
