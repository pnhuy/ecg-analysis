def set_seed(seed=0):
    import numpy, torch, random
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)