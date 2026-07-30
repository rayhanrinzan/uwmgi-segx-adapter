# UW-Madison SegX Offline Training

## On the cluster

```bash
cd /athena/imedslab/rar4037/uwmgi-segx-adapter
git pull origin main
chmod +x offline/*.sh
bash offline/build_usb_payload.sh
```

The finished folder is:

```text
/athena/imedslab/rar4037/uwmgi_usb_payload
```

Copy that entire folder to an exFAT USB.

## On the offline Ubuntu PC

Open Terminal in the USB's `uwmgi_usb_payload` folder and run:

```bash
bash START_HERE.sh
```

This installs everything into:

```text
~/uwmgi-offline
```

### One-epoch test

```bash
cd ~/uwmgi-offline/uwmgi-segx-adapter
SEGX_EPOCHS=1 bash offline/train_local.sh
```

### Full 30-epoch fold-0 baseline

```bash
cd ~/uwmgi-offline/uwmgi-segx-adapter
SEGX_EPOCHS=30 SEGX_FOLD=0 SEGX_BATCH_SIZE=32 bash offline/train_local.sh
```

### View TensorBoard graphs

In a second terminal:

```bash
cd ~/uwmgi-offline/uwmgi-segx-adapter
bash offline/start_tensorboard.sh
```

Then open Firefox at:

```text
http://127.0.0.1:6006
```

### Copy results back to the USB

Replace the USB path below with the path shown by Ubuntu:

```bash
cd ~/uwmgi-offline/uwmgi-segx-adapter
bash offline/export_results_to_usb.sh /media/$USER/SEGX_USB
```

The exported folder contains checkpoints, predictions, logs, TensorBoard data, and W&B offline runs. The W&B runs can later be uploaded from an internet-connected computer with `wandb sync`.
