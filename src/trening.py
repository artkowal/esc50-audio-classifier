"""Pętla treningowa wspólna dla obu modeli (zajęcia 4).

Przykłady:
    python src/trening.py --model cnn
    python src/trening.py --model gru --lr 3e-4 --scheduler plateau
    python src/trening.py --model cnn --bez-augmentacji

Harmonogramy learning rate (--scheduler):
    cosine   rozgrzewka, potem spadek kosinusowy do 0 w ustalonej liczbie epok (tempo narzuca harmonogram)
    plateau  rozgrzewka, potem LR zmniejszany o połowę dopiero, gdy strata walidacyjna nie poprawia się
             przez --plateau-patience epok (ReduceLROnPlateau: każdy model uczy się we własnym tempie)

Ustawienia (decyzja z 2026-10-22, opcja A), takie same dla obu modeli:
    CrossEntropyLoss, AdamW (lr 1e-3, weight decay 1e-2), batch 32, maks. 100 epok,
    rozgrzewka LR przez 5 epok, potem spadek kosinusowy, Early Stopping po 20 epokach bez poprawy
    straty walidacyjnej, zapis najlepszego checkpointu (najniższa strata walidacyjna).
Zbiór testowy NIE jest tu używany. Ocena na teście jest w osobnym skrypcie (zajęcia 8).

Wyniki:
    models/<nazwa>.pt                        najlepszy checkpoint (wagi + konfiguracja)
    docs/wyniki/<nazwa>/historia.csv         strata, accuracy, F1, learning rate w każdej epoce
    docs/wyniki/<nazwa>/podsumowanie.json    najlepsza epoka, wyniki walidacji, czasy, liczba parametrów
    docs/figures/trening/<nazwa>.png         krzywe uczenia
"""
import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dane import ROOT
from modele import CNNModel, GRUModel, count_params
from potok import compute_mel_stats, make_loaders, stats_path

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "docs" / "wyniki"
FIG_DIR = ROOT / "docs" / "figures" / "trening"


def parse_args():
    p = argparse.ArgumentParser(description="Trening modelu CNN lub GRU na ESC-50")
    p.add_argument("--model", choices=["cnn", "gru"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    p.add_argument("--warmup", type=int, default=5, help="liczba epok rozgrzewki learning rate")
    p.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    p.add_argument("--plateau-patience", type=int, default=5, help="plateau: epoki bez poprawy przed zmniejszeniem LR")
    p.add_argument("--patience", type=int, default=20, help="Early Stopping: epoki bez poprawy")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--bez-augmentacji", dest="augment", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nazwa", default=None, help="nazwa eksperymentu (domyślnie z ustawień)")
    return p.parse_args()


def build_model(args, audio, idx, device):
    if args.model == "gru":
        return GRUModel()
    path = stats_path(args.val_fold, args.test_fold)      # statystyki normalizacji z treningu tej iteracji
    if path.exists():
        stats = torch.load(path)
    else:
        mean, std = compute_mel_stats(audio, idx["train"], device)
        stats = {"mean": mean, "std": std}
        torch.save(stats, path)
    return CNNModel(stats["mean"], stats["std"], use_specaug=args.augment)


def warmup_cosine(warmup, epochs):
    """Mnożnik learning rate dla danej epoki: liniowy wzrost przez `warmup` epok, potem spadek kosinusowy do 0."""
    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Jedna epoka. Z optymalizatorem: trening (z aktualizacją wag), bez: ocena (bez gradientów)."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for wave, y in loader:
            wave, y = wave.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(wave)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # ochrona przed „wybuchem” gradientu
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "acc": float((preds == targets).mean()),
        "f1": float(f1_score(targets, preds, average="macro")),
    }


def plot_history(hist, best_epoch, title, path):
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")
    panels = [("loss", "Strata (CrossEntropy)"), ("acc", "Accuracy")]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(hist["epoch"], hist[f"train_{key}"], color=blue, linewidth=2, label="trening")
        ax.plot(hist["epoch"], hist[f"val_{key}"], color=orange, linewidth=2, label="walidacja")
        ax.set_title(label, loc="left", fontsize=10, color=ink)
    axes[2].plot(hist["epoch"], hist["lr"], color=muted, linewidth=2)
    axes[2].set_title("Learning rate", loc="left", fontsize=10, color=ink)
    axes[0].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.axvline(best_epoch, color=muted, linestyle="--", linewidth=1)
        ax.set_xlabel("epoka", fontsize=9, color=muted)
        ax.grid(color=grid, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{title} (linia przerywana: najlepsza epoka {best_epoch})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True            # stały rozmiar wejścia: cuDNN dobiera najszybsze algorytmy
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = args.nazwa or (f"{args.model}_{'aug' if args.augment else 'bezaug'}_lr{args.lr:g}_{args.scheduler}"
                          f"_val{args.val_fold}_test{args.test_fold}")
    for d in (MODELS_DIR, RESULTS_DIR / name, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    loaders, audio, idx = make_loaders(args.val_fold, args.test_fold, augment=args.augment)
    model = build_model(args, audio, idx, device).to(device)
    total, trainable = count_params(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(args.warmup, args.epochs))
    else:
        # Rozgrzewkę ustawiamy ręcznie, a po niej działa ReduceLROnPlateau (LR x 0.5 przy braku poprawy)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.plateau_patience, min_lr=1e-6)

    print(f"Eksperyment: {name} | urządzenie: {device} | parametry: {trainable:,}".replace(",", " "))
    print(f"Trening: {len(idx['train'])} nagrań (foldy bez {args.val_fold} i {args.test_fold}), "
          f"walidacja: {len(idx['val'])} (fold {args.val_fold}), augmentacja: {'tak' if args.augment else 'nie'}\n")

    history, best_loss, best_epoch, wait = [], float("inf"), 0, 0
    ckpt_path = MODELS_DIR / f"{name}.pt"
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.scheduler == "plateau" and epoch <= args.warmup:      # rozgrzewka: liniowy wzrost LR
            for group in optimizer.param_groups:
                group["lr"] = args.lr * epoch / args.warmup
        lr = optimizer.param_groups[0]["lr"]
        t0 = time.perf_counter()
        tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        t_train = time.perf_counter() - t0
        va = run_epoch(model, loaders["val"], loss_fn, device)
        if args.scheduler == "cosine":
            scheduler.step()                                   # aktualizacja learning rate po każdej epoce
        elif epoch > args.warmup:
            scheduler.step(va["loss"])                         # plateau: reaguje na stratę walidacyjną

        history.append({"epoch": epoch, "lr": lr, "train_time_s": t_train,
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})

        improved = va["loss"] < best_loss
        if improved:                                           # zapis najlepszego checkpointu
            best_loss, best_epoch, wait = va["loss"], epoch, 0
            torch.save({"model_name": args.model, "state_dict": model.state_dict(), "config": vars(args),
                        "epoch": epoch, "val": va, "params": total}, ckpt_path)
        else:
            wait += 1

        print(f"Epoka {epoch:3d} | lr {lr:.1e} | trening: strata {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"walidacja: strata {va['loss']:.3f} acc {va['acc']:.3f} F1 {va['f1']:.3f} | "
              f"{t_train:.1f} s {'*' if improved else ''}")

        if wait >= args.patience:
            print(f"\nEarly Stopping: brak poprawy straty walidacyjnej od {args.patience} epok.")
            break

    total_time = time.perf_counter() - t_start

    # Przywrócenie najlepszego modelu i ponowna ocena na walidacji
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best = run_epoch(model, loaders["val"], loss_fn, device)

    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS_DIR / name / "historia.csv", index=False)
    summary = {
        "nazwa": name, "model": args.model, "augmentacja": args.augment, "parametry": total,
        "najlepsza_epoka": best_epoch, "liczba_epok": len(hist),
        "val_loss": round(best["loss"], 4), "val_acc": round(best["acc"], 4), "val_f1_macro": round(best["f1"], 4),
        "train_acc_w_najlepszej_epoce": round(hist.loc[best_epoch - 1, "train_acc"], 4),
        "sredni_czas_epoki_trening_s": round(hist["train_time_s"].mean(), 2),
        "laczny_czas_s": round(total_time, 1),
        "ustawienia": {k: v for k, v in vars(args).items() if k != "nazwa"},
    }
    (RESULTS_DIR / name / "podsumowanie.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_history(hist, best_epoch, f"{args.model.upper()} ({name}): krzywe uczenia", FIG_DIR / f"{name}.png")

    print(f"\nNajlepsza epoka: {best_epoch} z {len(hist)} | walidacja: acc {best['acc']:.3f}, "
          f"F1 macro {best['f1']:.3f}, strata {best['loss']:.3f}")
    print(f"Średni czas epoki (trening): {summary['sredni_czas_epoki_trening_s']} s | łącznie: {total_time / 60:.1f} min")
    print(f"Zapisano: {ckpt_path}\n          {RESULTS_DIR / name}\n          {FIG_DIR / f'{name}.png'}")
