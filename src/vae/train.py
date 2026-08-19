from utils.experiments import VaeExperiment

def main():
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
    experiment.train(1000)


if __name__ == "__main__":
    main()