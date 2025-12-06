import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR

from .model import GRUDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

#############################################################################
# Implementing CTC Loss with label smoothing
def ctc_with_label_smoothing(input, target, input_length, target_length, eps=0.1):
    """
    Args:
        input: Prediction tensor from model.forward(X, dayIdx)
        target: target sequences with shape (N,S) where S=target sequence length
        input_length: length of input with shape (N,)
        target_length: length of target with shape (N,)
        eps: smoothing coefficient
    """
    log_prob = torch.permute(input.log_softmax(2), [1,0,2])

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    std_loss = loss_ctc(log_prob, target, input_length, target_length)

    prob = torch.exp(log_prob)
    entropy = - (prob * log_prob).sum(dim = -1).mean()

    new_loss = ((1 - eps) * std_loss) + (eps * entropy)
    return new_loss

#############################################################################

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # Experiment with AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.05,
        weight_decay=args["l2_decay"],
    )

    ###################################################
    # Modified learning rate
    # Warm-up -> Hold -> Cosine decay
    num_steps = args["nBatch"]
    warmup_steps = num_steps / 20 # Results in step of 500
    decay_step = (0.7/1.0) * num_steps

    warmup = LinearLR(
            optimizer,
            start_factor = 1 / warmup_steps,
            total_iters = warmup_steps,
            )
    constant = ConstantLR(
            optimizer,
            factor = 1.0,
            total_iters = decay_step - warmup_steps,
            )
    cosine = CosineAnnealingLR(
            optimizer,
            T_max = num_steps - decay_step,
            eta_min = args["lrStart"] / 10
            )
    scheduler = SequentialLR(
            optimizer,
            schedulers = [warmup, constant, cosine],
            milestones = [warmup_steps, decay_step],
            )
    ###################################################

    # --train--
    testLoss = []
    testCER = []
    timePerBatch = []
    max_norm = 1

    ###########################################
    # Initializing variables for early stopping
    best_cer = float("inf") # Set to arbitrarily large number such that first cer is the best_cer
    patience = 5 # Number of batches (in hundreds) to compare to before stopping
    patience_counter = 0
    ###########################################

    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
        
        
        ####################################################################
        # Use the function for CTC Loss with label smoothing
        pred = model.forward(X, dayIdx)
        x_len_mod = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        loss = ctc_with_label_smoothing(pred, y, x_len_mod, y_len)
        loss = torch.sum(loss)
        #####################################################################
        
        """
        #########################################################
        # Implementation of intermediate CTC Loss
        pred_main,pred_inter = model.forward(X, dayIdx)
        x_len_mod = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        loss_main = ctc_with_label_smoothing(pred_main, y, x_len_mod, y_len)
        loss_main = torch.sum(loss_main)

        # Include CTC loss for intermediate layer
        loss_inter = ctc_with_label_smoothing(pred_inter, y, x_len_mod, y_len)
        loss_inter = torch.sum(loss_inter)

        # Use weights to control the contribution of each loss
        w = 0.3
        loss = (1-w) * loss_main + w * loss_inter
        #########################################################
        """

        #####################################################################
        # Apply gradient clipping
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        #####################################################################

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)

                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                timePerBatch.append((endTime - startTime) / 100)

                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, Time/Batch: {(endTime - startTime)/100:>7.3f}"
                )
                #####################################
                # Apply early stopping
                if cer < best_cer:
                    best_cer = cer
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping activated")
                    break
                ######################################
                print(f"Patience counter: {patience_counter}")
                startTime = time.time()
            """
            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            """
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)
            tStats["timePerBatch"] = np.array(timePerBatch)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model

@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()