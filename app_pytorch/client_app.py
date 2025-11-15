"""app-pytorch: A Flower / PyTorch client for HDFS logs."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from app_pytorch.task import NeuralLogTransformer, load_data, train_model, test

# -----------------------
# Flower ClientApp
# -----------------------
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local HDFS partition."""

    # Load the model with received weights
    embed_dim = 768       # match your features dimension
    num_classes = 2       # adjust if needed
    max_len = 75          # sequence length
    model = NeuralLogTransformer(embed_dim=embed_dim, num_classes=num_classes, max_len=max_len)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load local partition
    partition_file = context.node_config["partition-file"]  # each client has its .pt file
    trainloader, _ = load_data(partition_file)

    # Train the model
    local_epochs = context.run_config.get("local-epochs", 1)
    lr = msg.content.get("config", {}).get("lr", 0.01)
    train_model(model, trainloader, epochs=local_epochs, lr=lr, device=device)

    # Return updated model and metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {"num-examples": len(trainloader.dataset)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local HDFS partition."""

    # Load model with received weights
    embed_dim = 768
    num_classes = 2
    max_len = 75
    model = NeuralLogTransformer(embed_dim=embed_dim, num_classes=num_classes, max_len=max_len)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load local partition for evaluation
    partition_file = context.node_config["partition-file"]
    _, testloader = load_data(partition_file)

    # Evaluate the model
    eval_loss, eval_acc = test(
        model,
        testloader,
        device,
        file_name="metrics_client.json",
        server_round=1
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)