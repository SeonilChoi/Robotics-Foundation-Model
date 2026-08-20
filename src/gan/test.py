import argparse

from utils.experiments import GanExperiment

def main(logdir, epochs, stride):
    generator_kwargs = {
        "latent_dim": 100,
        "hidden_dims": [128],
        "output_dim": 784,
        "activation_fn": "relu",
    }
    discriminator_kwargs = {
        "features_dim": 784,
        "hidden_dims": [128],
        "activation_fn": "relu",
    }
    optimizer_kwargs = {
        "lr": 0.0001,
        "betas": (0.5, 0.999),
    }
    experiment = GanExperiment(generator_kwargs, discriminator_kwargs, "cuda", optimizer_kwargs, 16)
    experiment.test(epochs, stride, logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs/gan/20260814_105012")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    main(args.logdir, args.epochs, args.stride)