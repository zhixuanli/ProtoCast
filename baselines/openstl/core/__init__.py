from torch.optim import Adam
from timm.scheduler.cosine_lr import CosineLRScheduler

timm_schedulers = (CosineLRScheduler,)


def get_optim_scheduler(args, epoch, model, steps_per_epoch):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=epoch, lr_min=args.min_lr, warmup_t=args.warmup_epoch,
        warmup_lr_init=args.warmup_lr, k_decay=args.lr_k_decay, t_in_epochs=True,
    )
    return optimizer, scheduler, True
