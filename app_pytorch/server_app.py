"""app-pytorch: A Flower / PyTorch server for HDFS logs."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from app_pytorch.task import NeuralLogTransformer, load_centralized_dataset_from_pt, test

# Path to centralized HDFS test data
CENTRALIZED_PT_FILE = "all_hdfs_data.pt"  # replace with your actual .pt file path

# -----------------------
# Create ServerApp
# -----------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Initialize global model
    embed_dim = 768       # match your features dimension
    num_classes = 2       # adjust if needed
    max_len = 75          # sequence length
    global_model = NeuralLogTransformer(embed_dim=embed_dim, num_classes=num_classes, max_len=max_len)

    # Wrap model state in ArrayRecord
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.01}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final global model to disk
    print("\nSaving final global model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global model on central HDFS dataset."""

    # Load model weights
    embed_dim = 768
    num_classes = 2
    max_len = 75
    model = NeuralLogTransformer(embed_dim=embed_dim, num_classes=num_classes, max_len=max_len)
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load centralized test dataset
    test_dataloader = load_centralized_dataset_from_pt(CENTRALIZED_PT_FILE)

    # Evaluate global model
    test_loss, test_acc = test(
        model,
        test_dataloader,
        device,
        file_name="metrics_server.json",
        server_round=server_round
    )

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})