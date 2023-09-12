#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
import utils.model as nnmodel
import utils.data_utils as data_utils

import tqdm
import sys

sys.path.append("/PoseForecasters/")
import utils_pipeline

# ==================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

datapath_preprocessed = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
config = {
    "item_step": 2,
    "window_step": 2,
    "input_n": 50,
    "output_n": 25,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}

in_features = len(config["select_joints"]) * 3
dim_used = list(range(in_features))


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    return sequences


# ==================================================================================================

# def apply_dct(all_seqs, input_n, output_n, dct_used):
#     batched = []
#     for i in range(len(all_seqs)):
#         all_seq = all_seqs[i]

#         dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
#         dct_m_in = dct_m_in[0:dct_used, :]
#         print(dct_m_in.shape)

#         pad_idx = np.repeat([input_n - 1], output_n)
#         i_idx = np.append(np.arange(0, input_n), pad_idx)

#         # (30, 75) * (75, 45)
#         input_dct_seq = np.matmul(dct_m_in, all_seq[i_idx, :])
#         input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])

#         batched.append(input_dct_seq[0])

#     inputs = np.array(batched)
#     return inputs


def apply_dct2(all_seqs, input_n, output_n, dct_used):
    dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
    dct_m_in = dct_m_in[0:dct_used, :]

    pad_idx = np.repeat(input_n - 1, output_n)
    i_idx = np.append(np.arange(0, input_n), pad_idx)

    # (30, 75) * (75, n)
    inputs_t = all_seqs[:, i_idx, :].transpose(1, 0, 2).reshape(input_n + output_n, -1)

    input_dct_seq = np.matmul(dct_m_in, inputs_t)
    input_dct_seq = input_dct_seq.reshape([dct_used, -1, len(dim_used)]).transpose(
        1, 2, 0
    )

    return input_dct_seq


# ==================================================================================================


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    script_name = os.path.basename(__file__).split(".")[0]
    script_name = (
        script_name
        + "_in{:d}_out{:d}_dct{:d}_L{:d}_J{:d}_T{:d}_P{:.1f}".format(
            opt.input_n,
            opt.output_n,
            opt.dct_n,
            opt.num_stage,
            opt.J,
            opt.tree_num,
            opt.edge_prob,
        )
    )
    checkpoint_dir = "3D_dct{:d}_L{:d}_J{:d}_Wpg{:d}_Wp{:d}".format(
        opt.dct_n, opt.num_stage, opt.J, 4, 2
    )

    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n

    # upper body parts
    upJ = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    # lower body joints
    downJ = np.array([0, 1, 2, 3, 4, 5, 6])

    dim_up = np.concatenate((upJ * 3, upJ * 3 + 1, upJ * 3 + 2))
    dim_down = np.concatenate((downJ * 3, downJ * 3 + 1, downJ * 3 + 2))
    n_up = dim_up.shape[0]
    n_down = dim_down.shape[0]
    part_sep = (dim_up, dim_down, n_up, n_down)

    model = nnmodel.GCN(
        in_d=dct_n,
        hid_d=opt.linear_size,
        p_dropout=opt.dropout,
        num_stage=opt.num_stage,
        node_n=in_features,
        J=opt.J,
        part_sep=part_sep,
        W_pg=opt.W_pg,
        W_p=opt.W_p,
    )

    if is_cuda:
        model.cuda()

    print(
        ">>> total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = utils_pipeline.load_dataset(
        datapath_preprocessed, "train", config
    )
    esplit = "test" if "mocap" in datapath_preprocessed else "eval"

    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_preprocessed, esplit, config
    )
    dataset_test, dlen_test = utils_pipeline.load_dataset(
        datapath_preprocessed, "test", config
    )

    # dataset_test, dlen_test = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config
    # )
    # dataset_train, dlen_train = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config
    # )
    # dataset_eval, dlen_eval = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        # model_path_len = 'checkpoint/{}/'.format(checkpoint_dir) + 'ckpt_' + script_name + '_last.pth.tar'
        model_path_len = (
            "{}/".format(opt.ckpt + "/" + checkpoint_dir)
            + "ckpt_"
            + script_name
            + "_best.pth.tar"
        )
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location="cpu")
        start_epoch = ckpt["epoch"]
        err_best = ckpt["err"]
        lr_now = ckpt["lr"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

        # Load preprocessed datasets
        label_gen_test = utils_pipeline.create_labels_generator(
            dataset_test["sequences"], config
        )

        test_l, test_3d = test(
            label_gen_test,
            model,
            input_n=input_n,
            output_n=output_n,
            is_cuda=is_cuda,
            dim_used=dim_used,
            dct_n=dct_n,
            opt=opt,
            dlen=dlen_test,
        )

        print(test_l)
        print(test_3d)
        exit()

    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print("==========================")
        print(">>> epoch: {} | lr: {:.5f}".format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(["epoch"])

        label_gen_train = utils_pipeline.create_labels_generator(
            dataset_train["sequences"], config
        )
        label_gen_eval = utils_pipeline.create_labels_generator(
            dataset_eval["sequences"], config
        )
        label_gen_test = utils_pipeline.create_labels_generator(
            dataset_test["sequences"], config
        )

        # per epoch
        lr_now, t_l = train(
            label_gen_train,
            model,
            optimizer,
            lr_now=lr_now,
            max_norm=opt.max_norm,
            is_cuda=is_cuda,
            dim_used=dim_used,
            dct_n=dct_n,
            opt=opt,
            dlen=dlen_train,
            input_n=input_n,
            output_n=output_n,
        )
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ["lr", "t_l"])

        v_3d = val(
            label_gen_eval,
            model,
            is_cuda=is_cuda,
            dim_used=dim_used,
            dct_n=dct_n,
            opt=opt,
            dlen=dlen_eval,
            input_n=input_n,
            output_n=output_n,
        )

        print(t_l, v_3d)
        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ["v_3d"])

        test_l, test_3d = test(
            label_gen_test,
            model,
            input_n=input_n,
            output_n=output_n,
            is_cuda=is_cuda,
            dim_used=dim_used,
            dct_n=dct_n,
            opt=opt,
            dlen=dlen_test,
        )
        print(test_l, test_3d)
        ret_log = np.append(ret_log, [test_3d])
        head = np.append(head, ["test_3d"])

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            if not os.path.exists(opt.ckpt + "/" + checkpoint_dir):
                os.makedirs(opt.ckpt + "/" + checkpoint_dir)
            df.to_csv(
                opt.ckpt + "/" + checkpoint_dir + "/" + script_name + ".csv",
                header=head,
                index=False,
            )
        else:
            with open(
                opt.ckpt + "/" + checkpoint_dir + "/" + script_name + ".csv", "a"
            ) as f:
                df.to_csv(f, header=False, index=False)

        print(os.system("ls " + opt.ckpt + "/" + checkpoint_dir + "/"))

        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
        else:
            is_best = False
        file_name = [
            "ckpt_" + script_name + "_best.pth.tar",
            "ckpt_" + script_name + "_last.pth.tar",
        ]
        utils.save_ckpt(
            {
                "epoch": epoch + 1,
                "lr": lr_now,
                "err": test_3d[0],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path=opt.ckpt + "/" + checkpoint_dir,
            is_best=is_best,
            file_name=file_name,
        )


def train(
    data_loader,
    model,
    optimizer,
    lr_now=None,
    max_norm=True,
    is_cuda=False,
    dim_used=[],
    dct_n=15,
    opt=None,
    dlen=0,
    input_n=50,
    output_n=25,
):
    t_l = utils.AccumLoss()

    model.train()

    nbatch = opt.train_batch
    batch_size = nbatch

    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
        total=int(dlen / nbatch),
    ):
        if batch_size == 1:
            continue

        sequences_train = prepare_sequences(batch, nbatch, "input", device)
        sequences_gt = prepare_sequences(batch, nbatch, "target", device)

        all_seq = np.concatenate([sequences_train, sequences_gt], axis=1)
        input_dct_seq = apply_dct2(all_seq, input_n, output_n, dct_n)
        inputs = input_dct_seq
        inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
        all_seq = torch.from_numpy(all_seq.astype(np.float32)).to(device)

        outputs = model(inputs)

        loss = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        t_l.update(loss.item() * batch_size, batch_size)

    return lr_now, t_l.avg


def test(
    data_loader,
    model,
    input_n=50,
    output_n=25,
    is_cuda=False,
    dim_used=[],
    dct_n=15,
    opt=None,
    dlen=0,
):
    model.eval()
    nbatch = opt.test_batch

    N = 0
    t_l = 0
    eval_frame = list(range(output_n))
    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
            total=int(dlen / nbatch),
        ):
            n = nbatch
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)
            all_seq = np.concatenate([sequences_train, sequences_gt], axis=1)

            # input_dct_seq = apply_dct(all_seq, input_n, output_n, dct_n)
            input_dct_seq = apply_dct2(all_seq, input_n, output_n, dct_n)
            inputs = input_dct_seq
            inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)

            # print(input_dct_seq.shape, input_dct_seq2.shape)
            # assert np.allclose(input_dct_seq, input_dct_seq2)

            outputs = model(inputs)

            seq_len = input_n + output_n
            dim_used_len = len(dim_used)
            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = torch.from_numpy(idct_m).float().cuda()
            outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
            outputs_3d = (
                torch.matmul(idct_m[:, 0:dct_n], outputs_t)
                .transpose(0, 1)
                .contiguous()
                .view(-1, dim_used_len, seq_len)
                .transpose(1, 2)
            )

            pred_p3d = outputs_3d[:, input_n:].reshape([nbatch, output_n, -1, 3])
            targ_p3d = (
                torch.from_numpy(sequences_gt.astype(np.float32))
                .to(device)
                .reshape([nbatch, output_n, -1, 3])
            )

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += (
                    torch.mean(
                        torch.norm(
                            targ_p3d[:, j, :, :].contiguous().view(-1, 3)
                            - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                            2,
                            1,
                        )
                    ).item()
                    * n
                )

            N += n

    return t_l / N, t_3d / N


def val(
    data_loader,
    model,
    is_cuda=False,
    dim_used=[],
    dct_n=15,
    opt=None,
    dlen=0,
    input_n=50,
    output_n=25,
):
    t_3d = utils.AccumLoss()

    model.eval()
    nbatch = opt.test_batch

    with torch.no_grad():
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
            total=int(dlen / nbatch),
        ):
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)

            all_seq = np.concatenate([sequences_train, sequences_gt], axis=1)
            input_dct_seq = apply_dct2(all_seq, input_n, output_n, dct_n)
            inputs = input_dct_seq
            inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
            all_seq = torch.from_numpy(all_seq.astype(np.float32)).to(device)

            outputs = model(inputs)

            n, _, _ = all_seq.data.shape
            m_err = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dct_n, dim_used)
            t_3d.update(m_err.item() * n, n)

    return t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    stime = time.time()
    main(option)
    ftime = time.time()
    print("Script took {} seconds".format(int(ftime - stime)))
