import torch


def get_gaussian_noise(mu, sigma):
    return torch.distributions.normal.Normal(mu, sigma)


def get_poisson_noise(rate):
    return torch.distributions.poisson.Poisson(rate)


def get_noise(noise_type, noise_shape, bg_model):
    if noise_type == 'gaussian':
        distribution = get_gaussian_noise(bg_model, bg_model)
    elif noise_type == 'poisson':
        distribution = get_poisson_noise(bg_model)
    return distribution.sample(noise_shape)


def extract_noise_from_frame(frame, position):
    rounded_pos = position.round().to(int)
    extracted_box = frame[rounded_pos[0]-3:rounded_pos[0]+3, rounded_pos[1]-3:rounded_pos[1]+3]
    return extracted_box.mean()
