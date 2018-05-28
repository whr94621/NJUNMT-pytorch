from .optim import  Optimizer

class LearningRateScheduler(object):
    """ The base class of learning rate scheduler

    When writing a new scheduler, two functions should be implemented.
        - ```should_scheduler``` is used the return the condition to trigger the scheduler.
        - ```get_new_lr``` is the function to compute the new learning rate.
    """
    def __init__(self, optimizer, schedule_freq, min_lr=-1.0):
        """
        Args:
            optimizer: An instance of ```optim.Optimizer```
            schedule_freq: The interval the scheduler should be triggered
            min_lr: The minimum learning rate
        """
        self.optimizer = optimizer # type:Optimizer
        self.min_lr = min_lr
        self.schedule_freq = schedule_freq

    def should_scheduler(self, global_step, **kwargs):
        """Condition to schedule learning rate

        Whether to anneal learning rate given some criteria.
        """
        raise NotImplementedError

    def get_new_lr(self, old_lr, global_step, **kwargs):
        """Compute new learning rate given the old learning rate
        """
        raise NotImplementedError

    def step(self, global_step, **kwargs):

        if self.should_scheduler(global_step, **kwargs):
            new_lrs = []

            for old_lr in self.optimizer.get_lrate():
                new_lrs.append(self.get_new_lr(old_lr, global_step, **kwargs))

            self.optimizer.set_lrate(new_lrs)

            return True

        return False


class LossScheduler(LearningRateScheduler):
    """ Schedule learning rate according to loss on development set.

    This method is first introduced in ***Stronger Baselines for Trustable Results in Neural Machine Translation***,
    M. Denkowski, et al.
    """
    def __init__(self, optimizer, schedule_freq, max_patience, min_lr=-1.0, decay_scale=0.5, warmup_steps=-1):
        """
        Args:
            optimizer: An instance of ```optim.Optimizer```
            schedule_freq: The interval the scheduler should be triggered
            min_lr: The minimum learning rate
            max_patience: Int. If learning rate does not decrease within these steps, the scheduler
                          will be triggered.
            decay_scale: Positive float. The factor multiplied to the learning rate.
            warmup_steps: Int. The scheduler can not be triggered within these steps.
        """
        super().__init__(optimizer, schedule_freq, min_lr)

        self.max_patience = max_patience
        self.decay_scale = decay_scale
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        self._max_loss = 1e12
        self._bad_counts = 0

    def should_scheduler(self, global_step, **kwargs):

        loss = kwargs['loss']

        if global_step <= self.warmup_steps:
            return False

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
    """ Learning Rate Scheduling introduced by Noam Shazeer in Attention Is All You Need

    The learning rate is computed like:
        lrate = origin_lrate * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})
    where origin_lrate = optim.init_lr * d_model ** (-0.5).

    When using noam scheduler, the initial learning rate should be 1.0 as default, and smaller if fining-tuned
    is needed.
    """
    def __init__(self, optimizer, d_model, schedule_freq=1, warmup_steps=4000, min_lr=-1.0):
        """
        Args:
            optimizer: An instance of ```optim.Optimizer```
            schedule_freq: The interval the scheduler should be triggered. Default is 1
            min_lr: The minimum learning rate
            d_model: Int. The dimension of the model.
            warmup_steps: Int. The scheduler can not be triggered within these steps.
        """
        super().__init__(optimizer, schedule_freq, min_lr)

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def should_scheduler(self, global_step, **kwargs):
        return True

    def get_new_lr(self, old_lr, global_step, **kwargs):

        origin_lr = self.optimizer.init_lr * self.d_model ** (-0.5)

        new_lr = origin_lr * min(global_step ** (-0.5),
                                  global_step * self.warmup_steps ** (-1.5))

        return new_lr