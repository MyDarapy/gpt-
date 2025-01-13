import math

class GPTLearningRateScheduler:
    def __init__(self, max_lr: float, min_lr: float, warm_up_iters: int, max_iters: int, start_lr: float = 0.0):
        self.start_lr: float = start_lr
        self.min_lr: float = min_lr
        self.max_lr: float = max_lr
        self.warm_up_iters: int = warm_up_iters
        self.max_iters: int = max_iters

    def linear_warm_up(self, current_iter: int):
        return self.max_lr * (current_iter / self.warm_up_iters)

    def cosine_annealing(self, current_iter: int):
        decay_ratio = (current_iter - self.warm_up_iters) / (self.max_iters - self.warm_up_iters)
        assert 0 <= decay_ratio <= 1, f"decay ratio is not between 0 and 1"
        coeff = 1 + math.cos(math.pi * decay_ratio)
        return self.min_lr + (0.5 * coeff  * (self.max_lr - self.min_lr))

    def get_lr(self, current_iter: int):
        if current_iter <= self.warm_up_iters:
            lr = self.linear_warm_up(current_iter)
            return lr

        return self.cosine_annealing(current_iter)