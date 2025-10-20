from pathlib import Path

from einops import rearrange
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def val_epoch(cfg, dataset_v, model, loss_fn, acc_fn_v, callback_v):
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn = loss_fn
    pack.acc_fn_v = acc_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    is_img = True  # TODO XXX
    pack2 = Config({})
    # pack2.slotz = []
    """pack2.query = []  # TODO XXX
    if hasattr(pack2, "query"):
        def hook_fn(module, input, output):
            q_container.append(output.detach().clone())
        q_container = []
        hook_handle = model.m.initializ.register_forward_hook(hook_fn)
        hook_handle = model.m.transit.register_forward_hook(hook_fn)"""
    # ### <<<- experiment
    # global ts, te
    # pack.model.m.transit.ts_ = ts
    # pack.model.m.transit.te_ = te
    # ### ->>>

    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32"))
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32"))
    cnt = 0

    pack.isval = True
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(pack.dataset_v):
        # if i < 141 or i > 141:  # TODO XXX
        #     continue
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        """if hasattr(pack2, "query"):
            pack2.query.append(pt.stack(q_container, 1).half().cpu().numpy())
            q_container.clear()"""
        if hasattr(pack2, "slotz"):
            pack2.slotz.append(pack.output["slotz"].half().cpu().numpy())

        if 0:  # TODO XXX
            for image, segment in zip(pack.batch["image"], pack.output["segment2"]):
                pack.dataset_v.dataset.visualiz(
                    image, segment=segment, inorm=[mean, std], wait=0
                )

        if 0:  # TODO XXX
            # makdir
            save_dn = Path(cfg.name)
            if not Path(save_dn).exists():
                save_dn.mkdir(exist_ok=True)
            # read gt image and segment
            img_key = "image" if is_img else "video"
            imgs_gt = (  # image video
                (pack.batch[img_key] * std.cuda() + mean.cuda()).clip(0, 255).byte()
            )
            segs_gt = pack.batch["segment"].argmax(-1)  # onehot seg -> number seg
            # read pd attent -> pd segment
            if "segment2" in pack.output:
                segs_pd = pack.output["segment2"].argmax(-1)
            else:
                segs_pd = pack.output["segment"].argmax(-1)
            # visualize gt image or video
            for img_gt, seg_gt, seg_pd in zip(imgs_gt, segs_gt, segs_pd):
                if is_img:
                    img_gt, seg_gt, seg_pd = [  # warp img as vid
                        _[None] for _ in (img_gt, seg_gt, seg_pd)
                    ]
                for tcnt, (igt, sgt, spd) in enumerate(zip(img_gt, seg_gt, seg_pd)):
                    igt = igt.permute(1, 2, 0).cpu().numpy()
                    sgt = sgt.cpu().numpy()
                    spd = spd.cpu().numpy()
                    save_path = save_dn / f"{cnt:06d}-{tcnt:06d}"
                    cv2.imwrite(f"{save_path}-i.png", igt)
                    cv2.imwrite(
                        f"{save_path}-s.png", draw_segmentation_np(igt, sgt, alpha=0.9)
                    )
                    cv2.imwrite(
                        f"{save_path}-p.png", draw_segmentation_np(igt, spd, alpha=0.9)
                    )
                cnt += 1

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]

    for cb in pack.callback_v:
        if cb.__class__.__name__ == "AverageLog":
            pack2.log_info = cb.mean()
            break
        elif cb.__class__.__name__ == "HandleLog":
            pack2.log_info = cb.handle()
            break

    return pack2


def main_eval_single(
    # cfg_file="config-smoothsa/smoothsa_r_recogn-coco.py",  # 6680
    # ckpt_file="archive-recogn/smoothsa_r_recogn-coco/42/0002.pth",
    # cfg_file="config-spot/spot_r_recogn-coco.py",  # 6570
    # ckpt_file="archive-recogn/spot_r_recogn-coco/42/0002.pth",
    # cfg_file="config-smoothsa/smoothsav_r_recogn-ytvis.py",  # 8836
    # ckpt_file="archive-recogn/smoothsav_r_recogn-ytvis/42/0011.pth",
    # cfg_file="config-slotcontrast/slotcontrast_r_recogn-ytvis.py",  # 9151
    # ckpt_file="archive-recogn/slotcontrast_r_recogn-ytvis/42/0010.pth",
    # cfg_file="config-smoothsa/smoothsav_r-ytvis.py",
    # ckpt_file="../_20250620-dias0_randsfq_smoothsa-ckpt/20250620-dias0_randsfq_smoothsa-smoothsav-vvv/save/smoothsav_r-ytvis/42-0159.pth",
    cfg_file="config-slotcontrast/slotcontrast_r-ytvis.py",
    ckpt_file="../_20250620-dias0_randsfq_smoothsa-ckpt/20250620-dias0_randsfq_smoothsa-slotcontrast_ce/save/slotcontrast_r-ytvis/42-0155.pth",
):
    # data_dir = "/scratch/work/zhaor5/datasets"  # TODO XXX
    # data_dir = "/scratch/project_2008396/Datasets"
    # data_dir = os.environ["LOCAL_SCRATCH"]
    # print(f"data_dir: {data_dir}")
    data_dir = "/media/GeneralZ/Storage/Static/datasets"  # TODO XXX
    pt.backends.cudnn.benchmark = True

    cfg_file = Path(cfg_file)
    data_path = Path(data_dir)
    ckpt_file = Path(ckpt_file)

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)
    cfg.name = cfg_name

    ## datum init

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v // 2,  # TODO XXX
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    ## model init

    model = build_from_config(cfg.model)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        model.load(ckpt_file, None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()
    # model.compile()

    ## learn init

    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None  # TODO XXX change to current log file for eval
    callback_v = build_from_config(cfg.callback_v)

    ## do eval

    pack2 = val_epoch(cfg, dataload_v, model, loss_fn, acc_fn_v, callback_v)

    ## dump data

    if hasattr(pack2, "query"):
        query = np.concatenate(pack2.query, axis=0)  # (i*b,t,n,c)
        np.savez_compressed("query.npz", query)

    if hasattr(pack2, "slotz"):
        slotz = np.concatenate(pack2.slotz, axis=0)
        np.savez_compressed("slotz.npz", slotz)

    return pack2.log_info


def main_eval_multi():
    import os

    with open("eval_cfg.txt") as f:
        cfg_files0 = f.readlines()
    with open("eval_pth.txt") as f:
        ckpt_files0 = f.readlines()

    cfg_files = []
    for cfg_file0 in cfg_files0:
        cfg_file0 = cfg_file0[2:].strip()  # remove ./ and \n
        cfg_fn = cfg_file0.split("/")[-1].strip()
        # find cfg_fn in cfg_base_dir
        result = os.popen(f"find . -type f -path './config-*/{cfg_fn}'").read()
        result = result.strip().split("\n")
        assert len(result) == 1
        cfg_file = result[0]
        assert cfg_file.startswith("./config-") and cfg_file.endswith(".py")
        cfg_files.append(Path(cfg_file))

    ckpt_base_dir = Path(
        "/media/GeneralZ/Storage/Active/20250620-randsfq/_ckpt_vq_vfm_ocl_github"
    )
    ckpt_files = []
    for ckpt_file0 in ckpt_files0:
        ckpt_file0 = ckpt_file0[2:].strip()
        ckpt_file = ckpt_base_dir / ckpt_file0
        ckpt_files.append(ckpt_file)

    assert len(cfg_files) == len(ckpt_files)

    log_file = Path("eval_multi.csv")
    log_file.touch()
    keys = ("ari", "ari_fg", "mbo", "miou")
    # keys = ("recon", "align", "commit")
    for cfgf, ckptf in zip(cfg_files, ckpt_files):
        ckptn = ckptf.parent.name
        cname = ckptn[:-3]
        seed = int(ckptn[-2:])
        assert cname == cfgf.name[:-3]
        print(f"###\n{cname}\n###")
        print(cfgf.as_posix(), ckptf.as_posix())
        eval_info = main_eval_single(cfgf, ckptf)
        values = [eval_info[_] for _ in keys]
        values_str = ",".join([f"{_:.8f}" for _ in values])
        with open(log_file, "a") as f:
            f.writelines(f"{cname}-{seed},{values_str}\n")
    return


if __name__ == "__main__":
    # main_eval_single()
    main_eval_multi()
