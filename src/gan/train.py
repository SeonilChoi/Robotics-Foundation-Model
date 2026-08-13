from utils.experiments import GANExperiment

def main():
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
    experiment = GANExperiment(generator_kwargs, discriminator_kwargs, "cuda", optimizer_kwargs, 16)
    experiment.train(500, 2)


if __name__ == "__main__":
    main()