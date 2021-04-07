import torch

torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

def get_gaussian_noise(mu, sigma):
    return torch.distributions.normal.Normal(mu, sigma)


def get_poisson_noise(rate):
    return torch.distributions.poisson.Poisson(rate)


def get_noise(config):
    noise_shape = (config.frames, config.image_size, config.image_size)
    if config.noise_type == 'gaussian':
        distribution = get_gaussian_noise(config.bg_model, config.bg_model)
    elif config.noise_type == 'poisson':
        distribution = get_poisson_noise(config.bg_model)
    return distribution.sample(noise_shape)


# TODO: Implement this function later
def extract_noise_from_frame():
    pass
