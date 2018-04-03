import os
from collections import defaultdict

import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch.autograd import Variable
import numpy as np
from src.utils.common_utils import *
from src.utils.data_io import ZipDatasets, TextDataset, DataIterator
from src.metric.bleu_scorer import ExternalScriptBLEUScorer
import src.models
from src.models import *
from src.modules.criterions import NMTCritierion
from src.utils.optim import Optimizer
from src.utils.lr_scheduler import LossScheduler, NoamScheduler


def prepare_data(seqs_x, seqs_y=None, volatile=False, cuda=False, batch_first=True):
    def _np_pad_batch_2D(samples, pad, batch_first=True, volatile=False, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = Variable(torch.LongTensor(x_np).contiguous(),
                     volatile=volatile)
        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Vocab.BOS] + s + [Vocab.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Vocab.PAD,
                         volatile=volatile, cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [Vocab.BOS] + s + [Vocab.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Vocab.PAD,
                         volatile=volatile, cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    batch_dim=0,
                    shard_size=-1,
                    n_correctness=False
                    ):
    """
    :type model: Transformer

    :type critic: NMTCritierion
    """

    y_inp = seqs_y[:, :-1].contiguous()
    y_label = seqs_y[:, 1:].contiguous()

    dec_outs = model(seqs_x, y_inp)

    loss = critic(generator=model.generator,
                  shard_size=shard_size,
                  normalization=normalization,
                  batch_dim=batch_dim,
                  eval=eval,
                  dec_outs=dec_outs,
                  labels=y_label)

    if n_correctness:
        mask = y_label.data.ne(Vocab.PAD)
        pred = model.generator(dec_outs).data.max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.data.eq(pred).float().masked_select(mask).sum() / normalization

        return loss.data[0], num_correct

    return loss


def loss_validation(model, critic, valid_iterator):
    """
    :type model: Transformer

    :type critic: NMTCritierion

    :type valid_iterator: DataIterator
    """

    n_sents = 0
    n_tokens = 0.0

    sum_loss = 0.0
    sum_correct = 0.0

    model.eval()

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x, y = prepare_data(seqs_x, seqs_y, volatile=True, cuda=GlobalNames.USE_GPU)

        loss, num_correct = compute_forward(model=model,
                                            critic=critic,
                                            seqs_x=x,
                                            seqs_y=y,
                                            eval=True, n_correctness=True)

        if np.any(np.isnan(loss)):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_correct += num_correct

    return float(sum_loss / n_sents), float(sum_correct * 1.0 / n_tokens)


def bleu_validation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_tgt,
                    batch_size,
                    eval_at_char_level=False):
    """
    :type model: Transformer

    :type valid_iterator: DataIterator

    :type bleu_scorer: ExternalScriptBLEUScorer

    :type vocab_tgt: Vocab
    """

    def _split_into_chars(line):
        new_line = []
        for w in line:
            if vocab_tgt.token2id(w) not in {Vocab.UNK, Vocab.EOS, Vocab.BOS, Vocab.PAD}:
                # if not UNK, split into characters
                new_line += list(w)
            else:
                # if UNK, treat as a special character
                new_line.append(w)

        return new_line

    model.eval()

    trans = []

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seqs_x = batch[0]
        infer_progress_bar.update(len(seqs_x))

        x = prepare_data(seqs_x, volatile=True, cuda=GlobalNames.USE_GPU)

        word_ids = model(x, mode="infer", beam_size=5)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != Vocab.PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == Vocab.EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(' '.join(x_tokens))
            else:
                trans.append('%s' % vocab_tgt.id2token(Vocab.EOS))

    infer_progress_bar.close()

    # Merge bpe segmentation
    trans = [line.replace("@@ ", "") for line in trans]

    # Split into characters
    if eval_at_char_level is True:
        trans = [' '.join(_split_into_chars(line.strip().split())) for line in trans]

    if not os.path.exists("./valid"):
        os.mkdir("./valid")

    hyp_path = './valid/trans.iter{0}.txt'.format(uidx)

    with open(hyp_path, 'w') as f:
        for line in trans:
            f.write('%s\n' % line)

    bleu_v = bleu_scorer.corpus_bleu(hyp_path)

    return bleu_v


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

    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    saveto_collections = '%s.pkl' % os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_CHECKPOINIS_PREFIX)
    saveto_best_model = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    saveto_best_optim_params = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_OPTIMIZER_PARAMS_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocab(dict_path=data_configs['dictionaries'][0], max_n_words=data_configs['n_words'][0])
    vocab_tgt = Vocab(dict_path=data_configs['dictionaries'][1], max_n_words=data_configs['n_words'][1])

    train_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['train_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['bpe_codes'][0],
                    max_len=data_configs['max_len'][0]
                    ),
        TextDataset(data_path=data_configs['train_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['bpe_codes'][1],
                    max_len=data_configs['max_len'][1]
                    ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['valid_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['bpe_codes'][0]),
        TextDataset(data_path=data_configs['valid_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['bpe_codes'][1])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs['batch_size'],
                                     sort_buffer=training_configs['use_bucket'],
                                     sort_fn=lambda line: len(line[-1]))

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  sort_buffer=False)

    bleu_scorer = ExternalScriptBLEUScorer(reference_path=data_configs['bleu_valid_reference'],
                                           lang_pair=data_configs['lang_pair'])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    model_collections = Collections()

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    model_cls = model_configs.get("model")
    if model_cls not in src.models.__all__:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model_cls, src.models.__all__))

    nmt_model = eval(model_cls)(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    INFO(nmt_model)

    critic = NMTCritierion(num_tokens=vocab_tgt.max_n_words,
                           label_smoothing=model_configs['label_smoothing'])

    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=nmt_model,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # Whether Reloading model
    if FLAGS.reload is True:
        INFO('Reloading model...')
        timer.tic()
        if os.path.exists(saveto_best_model):
            INFO("Reloading model...")
            params = torch.load(saveto_best_model)
            nmt_model.load_state_dict(params)

            model_archives = Collections.unpickle(path=saveto_collections)
            model_collections.load(archives=model_archives)

            uidx = model_archives['uidx']
            bad_count = model_archives['bad_count']

            INFO("Done. Model reloaded.")

        if os.path.exists(saveto_best_optim_params):
            INFO("Reloading optimizer params...")
            optimizer_params = torch.load(saveto_best_optim_params)
            optim.optim.load_state_dict(defaultdict(dict, **optimizer_params))

            INFO("Done. Optimizer params reloaded.")

        INFO('Done. Elapsed time {0}'.format(timer.toc()))

    else:
        uidx = 0
        bad_count = 0

    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    if training_configs['decay_method'] == "loss":

        scheduler = LossScheduler(optimizer=optim,
                                  max_patience=training_configs['lrate_decay_patience'],
                                  min_lr=training_configs['min_lrate'],
                                  decay_scale=0.5
                                  )

    elif training_configs['decay_method'] == "noam":
        optim.init_lr = optimizer_configs['learning_rate'] * model_configs['d_model'] ** (-0.5)

        scheduler = NoamScheduler(optimizer=optim,
                                  warmup_steps=training_configs['decay_warmup_steps'])
    else:
        scheduler = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Prepare training

    params_best_loss = None

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    valid_loss = 1.0 * 1e12  # Max Float
    saving_files = []

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    for eidx in range(training_configs['max_epochs']):
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents"
                                     )
        for batch in training_iter:

            uidx += 1

            # ================================================================================== #
            # Learning rate annealing
            if np.mod(uidx, training_configs['decay_freq']) == 0 or FLAGS.debug:

                if scheduler.step(global_step=uidx, loss=valid_loss):

                    new_lr = list(optim.get_lrate())[0]
                    summary_writer.add_scalar("lrate", new_lr, global_step=uidx)

                    if training_configs['decay_method'] == "loss":
                        nmt_model.load_state_dict(params_best_loss)

            seqs_x, seqs_y = batch

            batch_size_t = len(seqs_x)
            cum_samples += batch_size_t
            cum_words += sum(len(s) for s in seqs_y)

            training_progress_bar.update(batch_size_t)

            # Prepare data
            x, y = prepare_data(seqs_x, seqs_y,
                                volatile=False,
                                cuda=GlobalNames.USE_GPU)

            #  Train for one batch
            nmt_model.train()
            # optim.zero_grad()
            nmt_model.zero_grad()
            loss = compute_forward(model=nmt_model,
                                   critic=critic,
                                   seqs_x=x,
                                   seqs_y=y,
                                   eval=False,
                                   normalization=batch_size_t,
                                   shard_size=training_configs['shard_size'])
            optim.step()

            # ================================================================================== #
            # Display some information
            if np.mod(uidx, training_configs['disp_freq']) == 0:
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                words_per_sen = cum_samples / (timer.toc(return_seconds=True))
                summary_writer.add_scalar("Speed (words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed (words/sen)", scalar_value=words_per_sen, global_step=uidx)

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0

            # ================================================================================== #
            # Saving checkpoints
            if np.mod(uidx, training_configs['save_freq']) == 0 or FLAGS.debug:

                INFO('Saving the model at iteration {}...'.format(uidx))

                if not os.path.exists(FLAGS.saveto):
                    os.mkdir(FLAGS.saveto)

                saveto_uidx = os.path.join(FLAGS.saveto, FLAGS.model_name + '.iter%d.tpz' % uidx)
                torch.save(nmt_model.state_dict(), saveto_uidx)

                Collections.pickle(path=saveto_collections,
                                   uidx=uidx,
                                   bad_count=bad_count,
                                   **model_collections.export())

                saving_files.append(saveto_uidx)

                INFO('Done')

                if len(saving_files) > 5:
                    for f in saving_files[:-1]:
                        os.remove(f)

                    saving_files = [saving_files[-1]]

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if np.mod(uidx, training_configs['loss_valid_freq']) == 0 or FLAGS.debug:

                valid_loss, valid_n_correct = loss_validation(model=nmt_model,
                                                              critic=critic,
                                                              valid_iterator=valid_iterator,
                                                              )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                summary_writer.add_scalar("n_correct", valid_n_correct, global_step=uidx)

                # If no bess loss model saved, save it.
                if len(model_collections.get_collection("history_losses")) == 0 or params_best_loss is None:
                    params_best_loss = nmt_model.state_dict()

                if valid_loss <= min_history_loss:
                    params_best_loss = nmt_model.state_dict()  # Export best variables

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if (np.mod(uidx, training_configs['bleu_valid_freq']) == 0 and uidx > training_configs['bleu_valid_warmup']) \
                    or FLAGS.debug:

                valid_bleu = bleu_validation(uidx=uidx,
                                             valid_iterator=valid_iterator,
                                             batch_size=training_configs['bleu_valid_batch_size'],
                                             model=nmt_model,
                                             bleu_scorer=bleu_scorer,
                                             eval_at_char_level=data_configs['eval_at_char_level'],
                                             vocab_tgt=vocab_tgt
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

                        # save optim params
                        INFO('Saving best optimizer params...')
                        best_optim_params = optim.optim.state_dict()
                        torch.save(best_optim_params, saveto_best_optim_params)

                        INFO('Done.')

                else:
                    bad_count += 1

                    if bad_count >= training_configs['early_stop_patience']:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                with open("./valid.txt", 'a') as f:
                    f.write("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}\n".format(uidx, valid_loss,
                                                                                                   valid_bleu, lrate,
                                                                                                   bad_count))

        training_progress_bar.close()


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
    vocab_src = Vocab(dict_path=FLAGS.source_dict_path, max_n_words=data_configs['n_words'][0])
    vocab_tgt = Vocab(dict_path=FLAGS.target_dict_path, max_n_words=data_configs['n_words'][1])

    valid_dataset = TextDataset(data_path=FLAGS.source_path,
                                vocab=vocab_src,
                                bpe_codes=FLAGS.source_bpe_codes)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  sort_buffer=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    model_cls = model_configs.get("model")
    if model_cls not in src.models.__all__:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model_cls, src.models.__all__))

    nmt_model = eval(model_cls)(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)

    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = torch.load(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        seqs_x = batch[0]

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, volatile=True, cuda=GlobalNames.USE_GPU)

        word_ids = nmt_model(x, mode="infer", beam_size=5)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != Vocab.PAD] for line in sent_t]
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
                if w == 0:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(' '.join(sample))
        translation.append(samples)

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(FLAGS.beam_size):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')
