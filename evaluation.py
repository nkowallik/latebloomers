# evaluation.py
import json
import argparse
import matplotlib.pyplot as plt
import time


def load_metrics(json_path: str):
    """Load metrics JSON where each key is a round and value is a dict of metrics."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # rounds come as strings "0","1",..., convert to sorted ints
    rounds = sorted(int(r) for r in data.keys())

    accuracies = []
    losses = []
    precisions = []
    recalls = []
    f1s = []

    for r in rounds:
        m = data[str(r)]
        accuracies.append(m.get("accuracy", None))
        losses.append(m.get("loss", None))
        precisions.append(m.get("precision", None))
        recalls.append(m.get("recall", None))
        f1s.append(m.get("f1", None))

    return rounds, accuracies, losses, precisions, recalls, f1s


def plot_metrics(
    rounds,
    accuracies,
    losses,
    precisions,
    recalls,
    f1s,
    output_path: str = "metrics_per_round.png",
):
    """Plot metrics over rounds and save to file."""
    plt.figure(figsize=(10, 6))

    # First subplot: loss and accuracy
    plt.subplot(2, 1, 1)
    plt.plot(rounds, losses, marker="o", label="Loss")
    plt.plot(rounds, accuracies, marker="o", label="Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Loss and Accuracy per Round")
    plt.grid(True)
    plt.legend()

    # Second subplot: precision, recall, f1
    plt.subplot(2, 1, 2)
    plt.plot(rounds, precisions, marker="o", label="Precision")
    plt.plot(rounds, recalls, marker="o", label="Recall")
    plt.plot(rounds, f1s, marker="o", label="F1-score")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 per Round")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics (accuracy, loss, precision, recall, f1) per round."
    )
    today = time.time()
    name = f"images/{today}-metrics.png"
    parser.add_argument(
        "--input",
        type=str,
        default="metrics_server.json",
        help="Path to metrics JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=name,
        help="Path to save output plot image.",
    )
    args = parser.parse_args()

    rounds, acc, loss, prec, rec, f1 = load_metrics(args.input)
    plot_metrics(rounds, acc, loss, prec, rec, f1, args.output)


if __name__ == "__main__":
    main()
