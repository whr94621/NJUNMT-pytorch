from collections import OrderedDict
from src.optim import Optimizer
from src.utils.common_utils import register

SCHEDULERS = {}


def register_sheduler(name: str):
    return register(name, SCHEDULERS)


class LearningRateScheduler(object):
    """ The base class of learning rate scheduler

    When writing a new scheduler, two functions should be implemented.
        - ```should_scheduler``` is used the return the condition to trigger the scheduler.
        - ```get_new_lr``` is the function to compute the new learning rate.
    """

    def __init__(self, optimizer, min_lr=-1.0):
        """
        Args:
            optimizer: An instance of ```optim.Optimizer```
            schedule_freq: The interval the scheduler should be triggered
            min_lr: The minimum learning rate
        """
        self.optimizer = optimizer
        self.min_lr = min_lr

        self._state = {}

    def update_lr(self, *args, **kwargs):
        """Compute new learning rate given the old learning rate
        """
        raise NotImplementedError

    def step(self, **kwargs):
        new_lrs = []

        for old_lr in self.optimizer.get_lrate():
            new_lrs.append(max(self.min_lr, self.update_lr(old_lr, **kwargs)))

        self.optimizer.set_lrate(new_lrs)

    def state_dict(self):
        state = OrderedDict()
        for name, value in self._state.items():
            state[name] = value

        return state

    def load_state_dict(self, state_dict, strict=True):

        for name in self._state.keys():
            if name not in state_dict:
                if strict:
                    print("Load mode is strict but {0} is not found in state_dict.".format(name))
                    raise KeyError
            else:
                self._state[name] = state_dict[name]


@register_sheduler("linear")
class LinearScheduler(LearningRateScheduler):

    def __init__(self, optimizer, decay_steps, min_lr):
        super(LinearScheduler, self).__init__(optimizer=optimizer, min_lr=min_lr)
        self.decay_steps = decay_steps

    def update_lr(self, old_lr, global_step, **kwargs):
        origin_lr = self.optimizer.init_lr

        new_lr = max(0, (origin_lr - self.min_lr) * (self.decay_steps - global_step) / self.decay_steps) + self.min_lr

        return new_lr


@register_sheduler("noam")
class NoamScheduler(LearningRateScheduler):

    def __init__(self, optimizer, d_model, warmup_steps, min_lr=-1.0):
        super(NoamScheduler, self).__init__(optimizer=optimizer, min_lr=min_lr)

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        # Update learning at first step
        self.step(global_step=1)

    def update_lr(self, old_lr, global_step, **kwargs):
        opt_corr = 0.002

        origin_lr = self.optimizer.init_lr * self.d_model ** (-0.5) * opt_corr * 5000.0

        new_lr = origin_lr * min(global_step ** (-0.5),
                                 global_step * self.warmup_steps ** (-1.5))

        return new_lr


@register_sheduler("loss")
class ReduceOnPlateauScheduler(LearningRateScheduler):

    def __init__(self, optimizer, patience, min_lr=-1.0, scale=0.5, mode="min"):

        super(ReduceOnPlateauScheduler, self).__init__(optimizer=optimizer, min_lr=min_lr)

        self.patience = patience
        self.scale = scale

        if mode not in {"min", "max"}:
            print("mode can only be 'min' or 'max'.")
            raise ValueError

        self.mode = mode

        self._state = dict()

        self._state["best"] = float("inf")

        if mode == "max":
            self.best = - self.best

        self._state["bad_count"] = 0

    def update_lr(self, old_lr, metric):

        if self.mode == "max":
            if metric > self._state["best"]:
                self._state["bad_count"] = 0
            else:
                self._state["bad_count"] += 1
        else:
            if metric < self._state["best"]:
                self._state["bad_count"] = 0
            else:
                self._state["bad_count"] += 1

        if self._state["bad_count"] == 0:
            self._state["best"] = metric
            return old_lr
        else:
            if self._state["bad_count"] <= self.patience:
                return old_lr
            else:
                new_lr = old_lr * self.scale
                self._state["bad_count"] = 0
                return new_lr


@register_sheduler("isqrt")
class InverseSqrtWithWarmupScheduler(LearningRateScheduler):
    """
    0~warmup_steps: linear increase. If warmup_steps is None, hold initial learning rate.
    warmup_steps~decay_steps: hold current learning rate.
    decay_steps~ : learning rate decay as an inverse sqrt function.
    """

    def __init__(self, optimizer, warmup_steps=None, decay_steps=None, min_lr=-1.0, init_lr=1e-7):

        super(InverseSqrtWithWarmupScheduler, self).__init__(optimizer=optimizer, min_lr=min_lr)

        if warmup_steps is None:
            warmup_steps = 1
        if decay_steps is None:
            decay_steps = float('inf')

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.init_lr = init_lr

        if self.warmup_steps > 1:
            self._bias = (self.init_lr * self.warmup_steps - self.max_lr) / (self.warmup_steps - 1)
        else:
            self._bias = 0.0

        self.step(global_step=1)

    @property
    def max_lr(self):
        return self.optimizer.init_lr

    def update_lr(self, old_lr, global_step, **kwargs):
        if global_step <= self.warmup_steps:
            return global_step * (self.max_lr - self._bias) / self.warmup_steps + self._bias
        elif self.warmup_steps < global_step <= self.decay_steps:
            return self.max_lr
        else:
            return self.max_lr * ((self.decay_steps / global_step) ** 0.5)


def build_scheduler(schedule_method: str, optimizer: Optimizer, scheduler_configs: dict):
    if schedule_method is None:
        return None
    elif schedule_method not in SCHEDULERS:
        raise KeyError("Unknown scheduler name {0}. Do not use lr_scheduling.".format(schedule_method))
    else:
        return SCHEDULERS[schedule_method](optimizer=optimizer, **scheduler_configs)
