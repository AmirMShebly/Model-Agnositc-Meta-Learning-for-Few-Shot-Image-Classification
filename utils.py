import torch

def inner_loop(model, optimizer, loss_fn, task_data, device):
    model.train()
    for data, labels in task_data:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

def outer_loop(model, optimizer, loss_fn, meta_data, device):
    for task_data in meta_data:
        inner_loop(model, optimizer, loss_fn, task_data, device)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    return accuracy
