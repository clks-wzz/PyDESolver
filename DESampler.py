#lucky
import random
class DESampler():
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_
        pass
    def sample(self):
        return self.min_ + random.random() * (self.max_ - self.min_)
    def force_var_inrange(self, var):
        if var >= self.min_ and var <= self.max_:
            return var
        else:
            return self.sample()
    
