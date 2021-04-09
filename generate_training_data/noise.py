import torch

torch.manual_seed(1234)


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


def extract_noise_from_frame(frame, position):
    rounded_pos = position.round().to(int)
    extracted_box = frame[rounded_pos[0]-3:rounded_pos[0]+3, rounded_pos[1]-3:rounded_pos[1]+3]
    return extracted_box.mean()
