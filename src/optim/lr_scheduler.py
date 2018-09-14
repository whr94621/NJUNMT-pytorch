from collections import OrderedDict


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


class NoamScheduler(LearningRateScheduler):

    def __init__(self, optimizer, d_model, warmup_steps, min_lr=-1.0):
        super(NoamScheduler, self).__init__(optimizer=optimizer, min_lr=min_lr)

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def update_lr(self, old_lr, global_step, **kwargs):
        opt_corr = 0.002

        origin_lr = self.optimizer.init_lr * self.d_model ** (-0.5) * opt_corr * 5000.0

        new_lr = origin_lr * min(global_step ** (-0.5),
                                 global_step * self.warmup_steps ** (-1.5))

        return new_lr


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
