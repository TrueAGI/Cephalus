import math
import random


class EpsilonGreedy:
    # Formula and default settings borrowed from
    # https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/

    def __init__(self, initial: float = 1.0, final: float = 0.01, log_decay: float = 0.995):
        assert initial >= final
        self.initial = initial
        self.final = final
        self.log_decay = log_decay

    def get_epsilon(self, steps: int) -> float:
        value = 1.0 - math.log10((steps + 1) * self.log_decay)
        return min(self.initial, max(value, self.final))

    def __call__(self, steps: int) -> bool:
        explore = random.random() < self.get_epsilon(steps)
        return explore
