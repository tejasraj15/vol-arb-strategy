import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
train_ratio = 0.60
batch_size = 16
max_epochs = 1000
lr = 1e-3
min_lr = 1e-4
dropout = 0.5
l2_rate = 1e-2
patience = 30
val_ratio = 0.15
random_seed = 42

CSV_PATH = "data/SPY.csv"
date_col = "Date"
close_col = "Adj Close"
return_col = "market_return"

#data loading
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Use pre-computed market_return if available, else compute from Adj Close
    if return_col in df.columns and df[return_col].notna().sum() > 100:
        df["ret"] = df[return_col]
    else:
        df["ret"] = df[close_col].pct_change()

    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    return df

"""
This function will help build all 16 components of the KS
"""
def build_HAR_components(df):
    ret = df["ret"].values.copy()
    ret_num = len(ret)

    RV = ret ** 2 #RV: Squared Daily return

    abs_ret = np.abs(ret)#absolute return
    BPV = np.concatenate([[0.0], abs_ret[:-1]*abs_ret[1:]]) #multiply consecutive returns, note: we append 0 at the front because the product would lead to n-1 values instead
    
    BPV_std = pd.Series(BPV).rolling(21, min_periods=5).std().values #computes std of BPV over 21 day windoe for each day
    BPV_std = np.nan_to_num(BPV_std, nan=1e-8) #any values with Nan in BPV_std can be replaced with a negligable number for data processing 
    BPV_std = np.where(BPV_std == 0, 1e-8, BPV_std) #any values with 0 in BPV_std can be replaced with a negligable number for data processing
    #we ensure std is not 0 or Nan as we dividde in jumps

    ABD_jump = np.maximum(RV-BPV, 0.0) #if RV is bigger than BPV, then a jump else if similar then 0
    ABD_CSP = RV - ABD_jump #RV is in two parts, the ABD_CSP and ABD_jump, CSP is the smooth variance after removing the jump


    BNS_jump = np.where(RV > 3.0 * BPV_std, ABD_jump,0.0) #only use ABD jump if RV exceeds 3 std of BPV else 0
    BNS_CSP = RV - BNS_jump #RV is in two parts, the BNS_CSP and BNS_jump, CSP is the smooth variance after removing the jump

    # Do Note: difference here is that we check ret instead of RV as proposed in the paper
    Jo_jump = np.where(np.abs(ret)>2.0*BPV_std, ABD_jump, 0.0) #only use ABD jump if absolute return exceeds 2 std of BPV else 0    
    Jo_CSP = RV-Jo_jump #RV is in two parts, the Jo_CSP and Jo_jump, CSP is the smooth variance after removing the jump

    RS_positive = np.where(ret>=0, RV, 0.0) #Realised semi variance, note that the addition of both negative and positive semi variance gets the RV
    RS_negative = np.where(ret<0, RV, 0.0)

    SJ = RS_positive - RS_negative #Signed jump, the difference between positive and negative semi variance, if positive then more positive jumps than negative jumps and vice versa
    SJ_positive = np.where(SJ>0, SJ, 0.0)
    SJ_negative = np.where(SJ<0, SJ, 0.0)

    negative_RV = np.where(ret<0, RV, 0.0)#where daily retun is negative, the negative RV exists

    TQ = np.abs(ret) **(4.0/3.0) #Tripower Quarticity, a measure of estimating var of RV 

    KS_Components = pd.DataFrame({
        "RV": RV,
        "BPV": BPV,
        "ABD_jump": ABD_jump,
        "ABD_CSP": ABD_CSP,
        "BNS_jump": BNS_jump,
        "BNS_CSP": BNS_CSP,
        "Jo_jump": Jo_jump,
        "Jo_CSP": Jo_CSP,
        "RS_positive": RS_positive,
        "RS_negative": RS_negative,
        "ret": ret,
        "SJ": SJ,
        "SJ_positive": SJ_positive,
        "SJ_negative": SJ_negative,
        "negative_RV": negative_RV,
        "TQ": TQ
    }, index = df.index)
    return KS_Components

components_inorder = ["RV", "BPV", "ABD_jump", "ABD_CSP", "BNS_jump", "BNS_CSP", "Jo_jump", "Jo_CSP", "RS_positive", "RS_negative", "ret", "SJ", "SJ_positive", "SJ_negative", "negative_RV", "TQ"]#rows of the 16x16 image

#This is what the CNN will refer to later when processing 16x16 image
def build_labels(KS_components):
    RV = KS_components["RV"].values

    label = np.where(np.roll(RV, -1) < RV, 1, 0) #looks at tomorrow and today and if threshold met, then produces 1 or 0
    label[-1] = 0 #last day has no tomorrow 
    return label 

#note that the image is 16x16 where 16 rows ar e the HAR components above and the 16 columns are the components over different time windows

def compute_rolling_window(series, lags): #lag is time horizon, series is one column of one of the KS components
    length = len(series) 
    arr = np.zeros((length, len(lags))) #creates empty array of n rows(days) and 16 columns
    #we use 16 different horizons, and length n for now 
    #16 time horizons means 16 columns
    for i, lag in enumerate(lags):
        if lag == 1:
            arr[:, i] = series #window of 1 requires doing nothing, use the raw
        else:
            arr[:,i] = pd.Series(series).rolling(lag, min_periods=1).mean().values #for window of 2,3,5,10,21, we compute the rolling mean over the window size and fill in the array column by column

    return arr #essentially, go through every day per lag and do rolling window mean stuff and repeat for every lag 

lags = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def build_images(components):
    n = len(components) #number of days found in the KS_components df returned above
    images = np.zeros((n,16,16), dtype = np.float32) #creates empty array of n rows(days) and 16 columns and 16 layers (for the 16 components)

    for i, column in enumerate(components_inorder):
        windowed = compute_rolling_window(components[column].values, lags) #for each component, we compute the rolling window for each of the 16 lags and get a n by 16 array
        images[:,i,:] = windowed #we fill in the 16 layers of the image with the windowed data for each component, so we get a n by 16 by 16 array at the end
        #all days, just component i, all lags
        #n images of 2D 16 by 16 
    return images[:, np.newaxis, :, :] #we add a new axis for the channel dimension, so we get a n by 1 by 16 by 16 array at the end, which is what the CNN will take in as input
#CNN requires inpout in format of (batch, channels, height, width)

def normalise_images(images_train, images_test): #the images array will be split into 60%, 40% for training, testing

    images_train_num = images_train.shape[0] #first dimension is n which is number if images
    flat_train = images_train.reshape(images_train_num, -1)#goes from (n,1,16,16) format to (n,256)
    scaler = StandardScaler() #scaler object that will subtract mean and div std, this WILL DO THE SCALING OF ALL VALUES TO SAME FOR CNN
    flat_train_scaled = scaler.fit_transform(flat_train) #fit the scaler to the training data and transform it, so we get a (n,256) array of scaled values for the training data
    flat_test = images_test.reshape(images_test.shape[0], -1)
    flat_test_scaled = scaler.transform(flat_test)
    train_scaled = flat_train_scaled.reshape(images_train.shape) #goes back to (n,1,16,16) format for the CNN
    test_scaled = flat_test_scaled.reshape(images_test.shape)
    return  train_scaled.astype(np.float32), test_scaled.astype(np.float32)


class RVDataset(Dataset):
    def __init__(self, images, labels):
        self.x = torch.tensor(images, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        

        
class CNN_HAR_KS(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1,32,kernel_size = 3, padding = 1), #appllies 32 filters of size 3x3, should be output 16x16 due to padding, final output is (32,16,16)
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size = 3, padding = 1),#dtakes 32 channels instead of 1, output here is (64,16,16)
            nn.ReLU(),
            #note that 1 channel means just raw pixel values and 32 chanelles mean more filtered versions of the image
            nn.MaxPool2d(kernel_size=2, stride = 2), #halfs from 16x16 spatial dimension to 8x8 so (64,8,8)
            nn.Dropout(p=dropout) #prevents overfitting
#shape after all this is 64*8*8 

        )
        self.fc_block = nn.Sequential(
            nn.Flatten(), #flattens 64*8*8 into 1D vector 
            nn.Linear(64*8*8, 64), #dense layer has 4096 inputs to 64 outputs, learns which features are mosst useful for predicting vol direction
            nn.ReLU(), #activation layer
            nn.Linear(64,2),#final layer is 2 outputs, vol up or down for each
        ) 

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x #produces two numbers for vol up and down, higher number is the predicted

def train_model(model, train_loader, val_loader, max_epochs, lr, min_lr, l2, patience):
    #model is the CNN_HAR_KS model we defined, train_loader is training images and labels, val_loader is validation images, rest is all technical ML stuff like iterations, learning rate etc.
    model = model.to(device) #move model to GPU if available
    criterion = nn.CrossEntropyLoss()#loss func we are using
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = l2) #optimizer we are using, weight decay is L2 regularization to prevent overfitting
    #above adjusts model weights 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2, min_lr = min_lr) #learning rate scheduler that reduces learning rate if validation loss does not improve for a certain number of epochs (patience)
    #when val loss decreases for these many eopochs i.e. patience above, then half learning rate but never reduce below the min lr we set
    #essentially, if model is not improving then we reduce learning rate to find better weights to not overshoot


    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []} #val_acc is validation accuracy, correct predictions/total validation days
    for epoch in range(1, max_epochs+1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader: #y_batch is the true labels
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) #move data to GPU if available
            optimizer.zero_grad() #clears old gradients
            logits = model(X_batch) #passes image batch into CNN to get 2 scores per image
            loss = criterion(logits, y_batch) #compares prediced with true labels
            loss.backward() #computes gradients of loss with respect to model parameters
            optimizer.step() #updates weights 
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1) #takes the higher score of the two for each image to get predicted label
                correct += (preds == y_batch).sum().item() #compares predicted with true labels to count correct predictions
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        scheduler.step(val_loss) #updates learning rate based on validation loss i.e. how many epcohs in a row val_loss does not improve, this will have effects on adjusting weights in next epcoh training, val does not adjust weights
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} #scave copy of current models weights and biases
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping at epoch {epoch} with best val loss {best_val_loss:.4f}, train loss {train_loss:.4f}, and val_acc {val_acc:.4f}".format(epoch=epoch, best_val_loss=best_val_loss, train_loss=train_loss, val_acc=val_acc))
                break
        if epoch % 50 == 0:
            print("Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}".format(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc))

    model.load_state_dict(best_state)
    return history 

def evaluate(model, loader): #loader here is the test_loader to look at accuracy of predictions of  unseen data
    model.eval()
    softmax = nn.Softmax(dim=1) #converts the logits into probabilities
    #arrays with list of 1s and 0s; all_lables are true answers, all_preds are predictions and all_probs are probability of 1
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch) #pass images through CNN and get scores for ech image
            probs = softmax(logits)[:,1].cpu().numpy() #converts scores of class 1 to probabilites and move to cpu and convert to numpy
            preds = logits.argmax(dim=1).cpu().numpy()#takes higher score as prediction
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs)
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    youden = sensitivity + specificity - 1

    return {
        "accuracy":    accuracy,
        "auc":         auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden":      youden,
    }


#Plotting graphs: note that if val loss rises and train loss falls, then there is overfitting
def plot_training_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"],   label="Val loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val accuracy", color="green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved: training_history.png")


def plot_sample_images(images: np.ndarray, labels: np.ndarray, n: int = 4):
    #visualising the 16x16 HAR images
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        axes[i].imshow(images[i, 0], aspect="auto", cmap="RdYlGn")
        axes[i].set_title(f"Label: {'↓ vol' if labels[i] == 1 else '↑ vol'}")
        axes[i].set_xlabel("Lag window")
        axes[i].set_ylabel("HAR component")
        axes[i].set_xticks(range(16))
        axes[i].set_xticklabels([f"MA{l}" for l in lags], rotation=90, fontsize=6)
        axes[i].set_yticks(range(16))
        axes[i].set_yticklabels(components_inorder, fontsize=6)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print("Saved: sample_images.png")

def main():
    print("Loading data")
    df = load_data(CSV_PATH)

    print("Building HAR componnts")
    components = build_HAR_components(df)
    labels = build_labels(components)
    print("Labels and components constructed")
    print("Constructing 16x16 images")
    drop = 21
    valid_components = components.iloc[21:-1].reset_index(drop=True) #we drop the first 21 days as they have many NaN values due to rolling windows and we drop the last day as it has no label
    labels_valid = labels[21:-1]
    images = build_images(valid_components)
    print("Images shape{} and now {} samples of 16x16 greyscale".format(images.shape, images.shape[0]))
    print("Splitting into train and test sets")
    n = len(labels_valid)
    n_train = int(n*train_ratio)
    n_val = int(n_train*val_ratio)
    n_train_ = n_train-n_val
    x_train_raw = images[:n_train]
    x_test_raw = images[n_train:]
    y_train = labels_valid[:n_train]
    y_test = labels_valid[n_train:]

    x_train_scaled, x_test_scaled =normalise_images(x_train_raw, x_test_raw)
    x_train =x_train_scaled[:n_train_]
    x_val = x_train_scaled[n_train_:]
    y_train_ = y_train[:n_train_]
    y_val = y_train[n_train_:]
    y_train = y_train_

    print("Train : {}, Val: {}, Test: {}".format(len(y_train), len(y_val), len(y_test)))
    plot_sample_images(x_train, y_train, n=4)
    train_dataset = RVDataset(x_train, y_train)
    val_dataset = RVDataset(x_val, y_val)
    test_dataset = RVDataset(x_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Trainnig CNN-HAR-KS model")
    model = CNN_HAR_KS(dropout)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad) #done to check overfitting, number of weights_biases against the no of datapoints
    print(f"Model has {params_num} trainable parameters")
    print(model)
    history = train_model(model, train_loader, val_loader, max_epochs, lr, min_lr, l2_rate, patience)
    plot_training_history(history)
    print("Evaluating on test set")
    metrics = evaluate(model, test_loader)
    print(f"  │  Accuracy    : {metrics['accuracy']:.4f}                   │")
    print(f"  │  AUC         : {metrics['auc']:.4f}                   │")
    print(f"  │  Sensitivity : {metrics['sensitivity']:.4f}                   │")
    print(f"  │  Specificity : {metrics['specificity']:.4f}                   │")
    print(f"  │  Youden Index: {metrics['youden']:.4f}│")
    print("  └─────────────────────────────────────────┘")

    torch.save(model.state_dict(), "cnn_har_ks_weights.pth")
    print("Saved model weights to cnn_har_ks_weights.pth")
    return model, metrics

if __name__ == "__main__":
    main()