import argparse

from utils.experiments import VaeExperiment


def main(logdir, epochs, length):
    encoder_kwargs = {
        "features_dim": 784,
        "latent_dim": 2,
        "hidden_dims": [128],
        "activation_fn": "relu",
    }
    decoder_kwargs = {
        "latent_dim": 2,
        "hidden_dims": [128],
        "output_dim": 784,
        "activation_fn": "relu",
    }
    optimizer_kwargs = {
        "lr": 0.0001,
        "betas": (0.5, 0.999),
    }
    experiment = VaeExperiment(encoder_kwargs, decoder_kwargs, "cuda", optimizer_kwargs, 16)
    experiment.test(epochs, length, logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs/vae/20260819_153244")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--length", type=int, default=20)
    args = parser.parse_args()
    main(args.logdir, args.epochs, args.length)