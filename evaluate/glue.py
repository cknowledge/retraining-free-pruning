import torch
from datasets import load_metric
import os

from utils.arch import apply_neuron_mask
from dataset.glue import target_dev_metric


@torch.no_grad()
def eval_glue_acc(model, head_mask, neuron_mask, dataloader, task_name,prune_model_op_path):
    IS_STSB = model.num_labels == 1
    metric = load_metric("glue", task_name)

    model.eval()
    handles = apply_neuron_mask(model, neuron_mask)
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        if IS_STSB:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    # Save the model after applying the head and neuron mask
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of DataParallel
    # torch.save(model_to_save.state_dict(), os.path.join(prune_model_op_path, "pruned_model.pt"))

    # Export the model to ONNX format
    # torch.onnx.export(model_to_save, (head_mask, **batch),os.path.join(prune_model_op_path, "pruned_model.onnx"),opset_version=11)

    # print("Pruned model saved in:"+ prune_model_op_path) 
    for handle in handles:
        handle.remove()

    eval_results = metric.compute()
    target_metric = target_dev_metric(task_name)
    accuracy = eval_results[target_metric]
    return accuracy, model_to_save
