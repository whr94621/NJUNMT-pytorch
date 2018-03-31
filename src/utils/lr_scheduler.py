from .optim import  Optimizer

class LearningRateScheduler(object):

    def __init__(self, optimizer, min_lr=-1.0):

        self.optimizer = optimizer # type:Optimizer
        self.min_lr = min_lr

    def should_scheduler(self, **kwargs):
        """Condition to schedule learning rate

        Whether to anneal learning rate given some criteria.
        """
        raise NotImplementedError

    def get_new_lr(self, old_lr, global_step, **kwargs):
        """Compute new learning rate given the old learning rate
        """
        raise NotImplementedError

    def step(self, global_step, **kwargs):

        if self.should_scheduler(**kwargs):
            new_lrs = []

            for old_lr in self.optimizer.get_lrate():
                new_lrs.append(self.get_new_lr(old_lr, global_step, **kwargs))

            self.optimizer.set_lrate(new_lrs)

            return True

        return False


class LossScheduler(LearningRateScheduler):

    def __init__(self, optimizer, max_patience, min_lr=-1.0, decay_scale=0.5):

        super().__init__(optimizer, min_lr)

        self.max_patience = max_patience
        self.decay_scale = decay_scale
        self.min_lr = min_lr

        self._max_loss = 1e12
        self._bad_counts = 0

    def should_scheduler(self, loss, **kwargs):

        if loss < self._max_loss:
            self._max_loss = loss
            self._bad_counts = 0
            return False
        else:
            self._bad_counts += 1

        if self._bad_counts > self.max_patience:
            self._bad_counts = 0 # Update learing rate and reset the bad counts.

            return True

    def get_new_lr(self, old_lr, global_step, **kwargs):

        new_lr = old_lr * self.decay_scale

        new_lr = max(self.min_lr, new_lr)

        return new_lr

class NoamScheduler(LearningRateScheduler):

    def __init__(self, optimizer, warmup_steps=4000, min_lr=-1.0):

        super().__init__(optimizer, min_lr)

        self.warmup_steps = warmup_steps

    def should_scheduler(self, **kwargs):
        return True

    def get_new_lr(self, old_lr, global_step, **kwargs):

        origin_lr = self.optimizer.init_lr

        new_lr = origin_lr * min(global_step ** (-0.5),
                                  global_step * self.warmup_steps ** (-1.5))

        return new_lr