import os
import random
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding import beam_search, ensemble_beam_search
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.modules.criterions import NMTCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.common_utils import *
from src.utils.logging import *

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "collections" in state_dict:
        return state_dict["model"]
    return state_dict


def save_checkpoint(saveto_prefix,
                    global_step,
                    model,
                    optim,
                    lr_scheduler,
                    collections,
                    max_keeps=1,
                    **kwargs):
    """ Saving checkpoints

    Checkpoints will be saved as such format:
        saveto_prefix.ckpt.[global_step].model
        saveto_prefix.ckpt.[global_step].optim
        saveto_prefix.ckpt.[global_step].collections
    """
    saveto_dir = os.path.dirname(saveto_prefix)

    if not os.path.exists(saveto_dir):
        os.mkdir(saveto_dir)

    ckpt_list_path = saveto_prefix + ".checkpoints"

    if os.path.exists(ckpt_list_path):
        with open(ckpt_list_path) as f:
            ckpt_list = f.readlines()
        ckpt_list = [line.strip() for line in ckpt_list]
    else:
        ckpt_list = []

    ckpt_state_dict = dict()

    ckpt_state_dict["model"] = model.state_dict()
    ckpt_state_dict["optim"] = optim.state_dict()
    if lr_scheduler is not None:
        ckpt_state_dict["lr_scheduler"] = lr_scheduler.state_dict()
    ckpt_state_dict["collections"] = collections.state_dict()

    saveto_path = saveto_prefix + ".ckpt." + str(global_step)
    torch.save(ckpt_state_dict, saveto_path)

    ckpt_list.append(os.path.basename(saveto_path))

    if len(ckpt_list) > max_keeps:
        out_of_date_ckpt = ckpt_list.pop(0)
        os.remove(os.path.join(saveto_dir, out_of_date_ckpt))

    with open(ckpt_list_path, 'w') as f:
        f.write('\n'.join(ckpt_list))


def load_latest_checkpoint(saveto_prefix,
                           model,
                           optim,
                           lr_scheduler,
                           collections):
    saveto_dir = os.path.dirname(saveto_prefix)
    ckpt_list_path = saveto_prefix + ".checkpoints"

    if not os.path.exists(ckpt_list_path):
        return

    with open(ckpt_list_path) as f:
        ckpt_list = f.readlines()

    latest_ckpt = ckpt_list[-1].strip()
    latest_ckpt_path = os.path.join(saveto_dir, latest_ckpt)

    ckpt_state_dict = torch.load(latest_ckpt_path)

    model.load_state_dict(ckpt_state_dict['model'])
    optim.load_state_dict(ckpt_state_dict['optim'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt_state_dict['lr_scheduler'])
    collections.load_state_dict(ckpt_state_dict['collections'])


def split_shard(*inputs, split_size=1):
    if split_size <= 1:
        yield inputs
    else:

        lengths = [len(s) for s in inputs[-1]]  #
        sorted_indices = np.argsort(lengths)

        # sorting inputs

        inputs = [
            [inp[ii] for ii in sorted_indices]
            for inp in inputs
        ]

        # split shards
        total_batch = sorted_indices.shape[0]  # total number of batches

        if split_size >= total_batch:
            yield inputs
        else:
            shard_size = total_batch // split_size

            _indices = list(range(total_batch))[::shard_size] + [total_batch]

            for beg, end in zip(_indices[:-1], _indices[1:]):
                yield (inp[beg:end] for inp in inputs)


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False
                    ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """
    y_inp = seqs_y[:, :-1].contiguous()
    y_label = seqs_y[:, 1:].contiguous()

    words_norm = y_label.ne(PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()
        torch.autograd.backward(loss)
        return loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
        return loss.item()


def loss_validation(model, critic, valid_iterator):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0
    n_tokens = 0.0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

        loss = compute_forward(model=model,
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    return float(sum_loss / n_sents)


def bleu_validation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_tgt,
                    batch_size,
                    valid_dir="./valid",
                    max_steps=10,
                    beam_size=5,
                    alpha=-1.0
                    ):
    model.eval()

    numbers = []
    trans = []

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_nums = batch[0]
        numbers += seq_nums

        seqs_x = batch[1]

        infer_progress_bar.update(len(seqs_x))

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=model, beam_size=beam_size, max_steps=max_steps, src_seqs=x, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(vocab_tgt.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % vocab_tgt.id2token(EOS))

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    infer_progress_bar.close()

    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

    with open(hyp_path, 'w') as f:
        for line in trans:
            f.write('%s\n' % line)

    with open(hyp_path) as f:
        bleu_v = bleu_scorer.corpus_bleu(f)

    return bleu_v


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path != "":
        INFO("Loading pretrained model from {}".format(pretrain_path))
        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():
            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue
            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def default_configs(configs):
    configs["data_configs"].setdefault("bleu_valid_reference", "")

    configs["data_configs"].setdefault("num_refs", 1)

    configs["model_configs"].setdefault("label_smoothing", 0.0)

    configs["training_configs"].setdefault("seed", 1234)

    configs["training_configs"].setdefault("norm_by_words", False)

    configs["training_configs"].setdefault("buffer_size", 20 * configs["training_configs"]["batch_size"])

    configs["training_configs"].setdefault("update_cycle", 1)

    configs["training_configs"].setdefault("bleu_valid_beam_size", 4)

    configs["training_configs"].setdefault("bleu_valid_max_steps", 150)

    configs["training_configs"].setdefault("num_kept_checkpoints", 1)

    configs['training_configs'].setdefault("bleu_valid_alpha", 0.6)

    configs["training_configs"].setdefault("bleu_valid_configs", {
                                               "beam_size": 5,
                                               "alpha": 0.0,
                                               "max_steps": 150,
                                               "sacrebleu_args": "-tok none -lc",
                                               "postprocess": False
                                           })

    return configs


def train(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        CURRENT_DEVICE = "cpu"
    else:
        CURRENT_DEVICE = "cuda:0"

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)
    # Add default configs
    configs = default_configs(configs)
    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    GlobalNames.SEED = training_configs['seed']

    set_seed(GlobalNames.SEED)

    saveto_best_model = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
    train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True)

    bleu_scorer = SacreBLEUScorer(reference_path=data_configs["bleu_valid_reference"],
                                  num_refs=data_configs["num_refs"],
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial collections
    model_collections = Collections()

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    INFO(nmt_model)

    critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'])

    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, FLAGS.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=nmt_model,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params']
                      )
    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if FLAGS.reload:
        load_latest_checkpoint(saveto_prefix=os.path.join(FLAGS.saveto, FLAGS.model_name),
                               model=nmt_model,
                               optim=optim,
                               lr_scheduler=scheduler,
                               collections=model_collections)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float
    saving_files = []

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents"
                                     )
        for batch in training_iter:

            uidx += 1

            if scheduler is None:
                pass
            elif optimizer_configs["schedule_method"] == "loss":
                scheduler.step(metric=best_valid_loss)
            else:
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)

            cum_samples += n_samples_t
            cum_words += n_words_t

            training_progress_bar.update(n_samples_t)

            optim.zero_grad()

            # Prepare data
            for seqs_x_t, seqs_y_t in split_shard(seqs_x, seqs_y, split_size=training_configs['update_cycle']):
                x, y = prepare_data(seqs_x_t, seqs_y_t, cuda=GlobalNames.USE_GPU)

                loss = compute_forward(model=nmt_model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       eval=False,
                                       normalization=n_samples_t,
                                       norm_by_words=training_configs["norm_by_words"])
            optim.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=FLAGS.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

                if not is_early_stop:
                    save_checkpoint(saveto_prefix=os.path.join(FLAGS.saveto, FLAGS.model_name),
                                    global_step=uidx,
                                    model=nmt_model,
                                    optim=optim,
                                    lr_scheduler=scheduler,
                                    collections=model_collections,
                                    max_keeps=training_configs["num_kept_checkpoints"])

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=FLAGS.debug):
                valid_loss = loss_validation(model=nmt_model,
                                             critic=critic,
                                             valid_iterator=valid_iterator,
                                             )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                best_valid_loss = min_history_loss

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
                                       every_n_step=training_configs['bleu_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=FLAGS.debug):

                valid_bleu = bleu_validation(uidx=uidx,
                                             valid_iterator=valid_iterator,
                                             batch_size=training_configs["bleu_valid_batch_size"],
                                             model=nmt_model,
                                             bleu_scorer=bleu_scorer,
                                             vocab_tgt=vocab_tgt,
                                             valid_dir=FLAGS.valid_path,
                                             max_steps=training_configs["bleu_valid_configs"]["max_steps"],
                                             beam_size=training_configs["bleu_valid_configs"]["beam_size"],
                                             alpha=training_configs["bleu_valid_configs"]["alpha"]
                                             )

                model_collections.add_to_collection(key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())

                summary_writer.add_scalar("bleu", valid_bleu, uidx)
                summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)

                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        INFO('Saving best model...')

                        # save model
                        best_params = nmt_model.state_dict()
                        torch.save(best_params, saveto_best_model)
                        INFO('Done.')

                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count
                ))

        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, buffer_size=100000, numbering=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=nmt_model, beam_size=FLAGS.beam_size, max_steps=FLAGS.max_steps,
                                   src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        result_numbers += numbers

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == vocab_tgt.EOS:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(vocab_tgt.tokenizer.detokenize(sample))
        translation.append(samples)

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')


def ensemble_translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, buffer_size=100000, numbering=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_models = []

    model_path = FLAGS.model_path

    for ii in range(len(model_path)):

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
        nmt_model.eval()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        INFO('Reloading model parameters...')
        timer.tic()

        params = load_model_parameters(model_path[ii], map_location="cpu")

        nmt_model.load_state_dict(params)

        if GlobalNames.USE_GPU:
            nmt_model.cuda()

        nmt_models.append(nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = ensemble_beam_search(nmt_models=nmt_models, beam_size=FLAGS.beam_size, max_steps=FLAGS.max_steps,
                                            src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == vocab_tgt.EOS:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(vocab_tgt.tokenizer.detokenize(sample))
        translation.append(samples)

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')
