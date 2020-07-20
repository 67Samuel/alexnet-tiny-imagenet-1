# making SNIP
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, img_size=None, num_channels=3):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    if type(img_size) == int:
        inputs = inputs.view(-1,num_channels,img_size,img_size).float().requires_grad_()
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
    
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
    
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)

def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask
            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask)) #hook masks onto respective weights

# to aid in counting percentage of layers snipped        
def count_zeros(model):
    total_item_count = 0
    total_zero_count = 0
    layer_zero_count_dict = {}
    layer_item_count_dict = {}
    for num_params,_ in enumerate(model.parameters()):
        layer_zero_count_dict[num_params] = 0
        layer_item_count_dict[num_params] = 0
    
    for n, layer in enumerate(model.parameters()):
        try:
            for channel in layer:
                for kernal_2d in channel:
                    for kernal_1d in kernal_2d:
                        for item in kernal_1d:
                            if item.detach() == 0:
                                layer_zero_count_dict[n] += 1
                                total_zero_count += 1
                            layer_item_count_dict[n] += 1
                            total_item_count += 1
        except TypeError:
            # case where channel is a single valued tensor (bias)
            try:
                if channel.detach() == 0:
                    layer_zero_count_dict[n] += 1
                    total_zero_count += 1
                layer_item_count_dict[n] += 1
                total_item_count += 1
            except RuntimeError:
                # case where channel is a multi-valued tensor (bias)
                for item in channel:
                    if item.detach() == 0:
                        layer_zero_count_dict[n] += 1
                        total_zero_count += 1
                    layer_item_count_dict[n] += 1
                    total_item_count += 1
    return total_zero_count, total_item_count, layer_zero_count_dict, layer_item_count_dict
    
def percentage_snipped(original_model, snipped_model):
    _, _, original_model_perlayer_zero_count_dict, layer_item_count_dict = count_zeros(original_model)
    _, _, snipped_model_perlayer_zero_count_dict, _ = count_zeros(snipped_model)
    snipped_perlayer_zero_count_dict = {}
    percentage_snipped_perlayer_dict = {}
    for n, layer in enumerate(original_model_perlayer_zero_count_dict):
        key_name = f"layer {n}"
        snipped_perlayer_zero_count_dict[n] = snipped_model_perlayer_zero_count_dict[n] - original_model_perlayer_zero_count_dict[n] 
        percentage_snipped_perlayer_dict[key_name] = (snipped_perlayer_zero_count_dict[n]/layer_item_count_dict[n])*100
    return percentage_snipped_perlayer_dict
