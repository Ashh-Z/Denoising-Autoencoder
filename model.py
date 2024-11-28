import torch
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
from numerapi import NumerAPI
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from copy import deepcopy

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# Download and prepare the data
napi = NumerAPI()
DATA_VERSION = "v5.0"

# Download the feature metadata file
napi.download_dataset(f"{DATA_VERSION}/features.json")

# Read the metadata
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_sets = feature_metadata["feature_sets"]
feature_set = feature_sets["medium"]

# Download the training data
napi.download_dataset(f"{DATA_VERSION}/train.parquet")

# Load only the "medium" feature set
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era", "target"] + feature_set
)

train = train[train["era"].isin(train["era"].unique()[::])]

feature_names = [i for i in train.columns.tolist() if i != 'target' and i != 'era']
print(f'Number of features: {len(feature_names)}')

# Targets
targets = ['target']

# Prepare the dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_decoder, y_ae_targets, y_targets):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_decoder = torch.tensor(y_decoder, dtype=torch.float32)
        self.y_ae_targets = torch.tensor(y_ae_targets, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx],
                self.y_decoder[idx],
                self.y_ae_targets[idx],
                self.y_targets[idx])

class CustomModel(nn.Module):
    def __init__(self, num_columns, num_labels, hidden_units, dropout_rates):
        super(CustomModel, self).__init__()
        self.num_columns = num_columns
        self.num_labels = num_labels
        self.hidden_units = hidden_units

        # Input batch normalization
        # self.batch_norm0 = nn.BatchNorm1d(num_features=num_columns)
        self.layer_norm0 = nn.LayerNorm(normalized_shape=num_columns)


        # Encoder
        self.gaussian_noise_std = dropout_rates[0]
        self.encoder_dense1 = nn.Linear(num_columns, hidden_units[0]) # 705 -> 512
        self.encoder_ln1 = nn.LayerNorm(normalized_shape=hidden_units[0]) # self.encoder_bn1 = nn.BatchNorm1d(hidden_units[0])
        self.encoder_activation = nn.SiLU()
        self.encoder_dense2 = nn.Linear(hidden_units[0], hidden_units[1]) # 512 -> 224  
        self.encoder_ln2 = nn.LayerNorm(normalized_shape=hidden_units[1]) #self.encoder_bn2 = nn.BatchNorm1d(hidden_units[1])

        # Decoder
        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder_dense1 = nn.Linear(hidden_units[1], num_columns) # 224 -> 705

        # x_ae
        self.x_ae_dense = nn.Linear(num_columns, hidden_units[2]) # 705 -> 224
        self.x_ae_ln = nn.LayerNorm(normalized_shape=hidden_units[2]) # self.x_ae_bn = nn.BatchNorm1d(hidden_units[2])
        self.x_ae_activation = nn.SiLU()
        self.x_ae_dropout = nn.Dropout(dropout_rates[2])

        # out_ae
        self.out_ae_dense = nn.Linear(hidden_units[2], num_labels) # final mlp regression layer : 224 -> 1

        # out_encoder
        self.out_encoder = nn.Linear(hidden_units[2], num_labels)

        # x concatenation
        concat_size = num_columns + hidden_units[1] #  concat_szie = 705 + 224 = 929 
        self.x_ln = nn.LayerNorm(normalized_shape=concat_size) # self.x_bn = nn.BatchNorm1d(concat_size)
        self.x_dropout = nn.Dropout(dropout_rates[3])

        # Layers in the loop
        self.dense_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(3, len(hidden_units)):
            """ 
                i = 3 : 
                    929 -> 784
                    batchnorm(784)
                    dropout(0.3)
                i = 4 : 
                    784 -> 512 
                    batchnorm(512)
                    dropout(0.25)
                i = 5 :
                    512 -> 224
                    batchnorm(224)
                    dropout(0.4)  
            """
            input_size = concat_size if i == 3 else hidden_units[i - 1]
            output_size = hidden_units[i]
            self.dense_layers.append(nn.Linear(input_size, output_size))
            self.ln_layers.append(nn.LayerNorm(normalized_shape=output_size)) # self.bn_layers.append(nn.BatchNorm1d(output_size))
            self.activation_layers.append(nn.SiLU())
            self.dropout_layers.append(nn.Dropout(dropout_rates[i + 2]))
            print(f'i: {i} | input size: {input_size} | output size: {output_size} | droput: {dropout_rates[i+2]}')

        # out
        self.out_dense = nn.Linear(hidden_units[-1], num_labels)

    def forward(self, x):
        x0 = self.layer_norm0(x)

        # Apply Gaussian noise to x0 during training
        if self.training and self.gaussian_noise_std > 0:
            noise = torch.randn_like(x0) * self.gaussian_noise_std
            x_noisy = x0 + noise
        else:
            x_noisy = x0

        # Encoder
        encoder = self.encoder_dense1(x_noisy) # -> 512
        encoder = self.encoder_ln1(encoder)
        encoder = self.encoder_activation(encoder)
        encoder = self.encoder_dense2(encoder) # -> 256
        encoder = self.encoder_ln2(encoder)
        encoder = self.encoder_activation(encoder)        

        # out encoder 
        out_enc = self.out_encoder(encoder)
        out_enc = torch.sigmoid(out_enc)

        # Decoder
        decoder = self.decoder_dropout(encoder)
        decoder = self.decoder_dense1(decoder)
        # decoder = self.decoder_bn1(decoder)
        # decoder = self.decoder_activation(decoder)
        decoder = self.decoder_dropout(decoder)
        # 'decoder' output is the reconstruction of the inputs

        # x_ae
        x_ae = self.x_ae_dense(decoder)
        x_ae = self.x_ae_ln(x_ae)
        x_ae = self.x_ae_activation(x_ae)
        x_ae = self.x_ae_dropout(x_ae)

        # out_ae
        out_ae = self.out_ae_dense(x_ae)
        out_ae = torch.sigmoid(out_ae)

        # x concatenation
        x_concat = torch.cat([x0, encoder], dim=1)
        x_concat = self.x_ln(x_concat)
        x_concat = self.x_dropout(x_concat)

        x = x_concat

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.ln_layers[i](x)
            x = self.activation_layers[i](x)
            x = self.dropout_layers[i](x)

        out = self.out_dense(x)
        out = torch.sigmoid(out)

        return out_enc, decoder, out_ae, out

params = {
    "num_columns": len(feature_names),
    "num_labels": len(targets),
    # "hidden_units": [512, 256, 896, 448, 448, 256],
    "hidden_units": [512, 224, 224, 784, 512, 224],
    "dropout_rates": [
        0.05, ## this is sigma for gaussian noise 
        0.035,
        0.4,
        0.1,
        0.4,
        0.3,
        0.25,
        0.4
    ],
    "lr": 1e-4,
}

model = CustomModel(
    num_columns=params['num_columns'],
    num_labels=params['num_labels'],
    hidden_units=params['hidden_units'],
    dropout_rates=params['dropout_rates'],
).to(device)

# Define optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
criterion_decoder = nn.MSELoss()
criterion_ae_targets = nn.MSELoss()
criterion_targets = nn.MSELoss()
criterion_enc_targets = nn.MSELoss()

# Hyperparameters for MER
alpha = params['lr']  # Learning rate for SGD updates
beta = 0.1    # Within-batch meta-learning rate
gamma = 0.1   # Across-batch meta-learning rate
s = 5         # Number of batches sampled per data point
k = 10        # Number of SGD steps per batch
buffer_size = 1000  # Size of the memory buffer
batch_size = 1    # Set batch_size to 512

memory_buffer = []
age = 0  # Keeps track of the number of data points seen

def update_memory_buffer(memory_buffer, buffer_size, age, X_batch, y_batch):
    batch_size_actual = X_batch.size(0)
    for i in range(batch_size_actual):
        xi = X_batch[i]
        yi = y_batch[i]
        if len(memory_buffer) < buffer_size:
            memory_buffer.append((xi, yi))
        else:
            p = random.randint(0, age - 1)
            if p < buffer_size:
                memory_buffer[p] = (xi, yi)
        age += 1
    return memory_buffer, age

# Open the log file
log_file = open('mer1_training_losses.txt', 'w')
log_file.write('Era\tTotal Loss\tDecoder Loss\tAE_target Loss\tTarget Loss\tEncoder Pred Loss\n')
log_file.close()

# Training loop
eras_val = sorted(train['era'].unique())
print(f"Total eras: {len(eras_val)}")

for era_idx, era in enumerate(eras_val, 1):
    print(f'\nTraining on era: {era}')
    current_train_subset = train[train['era'] == era]
    dataset = CustomDataset(
        X=current_train_subset[feature_names].values,
        y_decoder=current_train_subset[feature_names].values,
        y_ae_targets=current_train_subset[targets].values,
        y_targets=current_train_subset[targets].values,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize loss accumulators
    total_loss = 0.0
    running_loss_decoder = 0.0
    running_loss_ae_targets = 0.0
    running_loss_targets = 0.0
    running_loss_enc_pred = 0.0

    for batch in tqdm(dataloader):
        batch_size_actual = batch[0].size(0)  # In case the last batch is smaller
        age += batch_size_actual  # Increment the age counter by actual batch size

        # Get the current batch of data
        X_batch, y_decoder_batch, y_ae_targets_batch, y_targets_batch = batch
        X_batch = X_batch.to(device)
        y_decoder_batch = y_decoder_batch.to(device)
        y_ae_targets_batch = y_ae_targets_batch.to(device)
        y_targets_batch = y_targets_batch.to(device)

        # Save θ_A0
        theta_A0 = deepcopy(model.state_dict())

        # Initialize θ_A_s with θ_A0
        theta_A_s = deepcopy(theta_A0)

        # Within-batch updates
        for i in range(s):
            # Save θ_W_i0
            theta_W_i0 = deepcopy(model.state_dict())

            # SGD updates
            for j in range(k):
                # Sample from memory buffer
                if len(memory_buffer) > 0:
                    mem_batch_size = min(len(memory_buffer), batch_size_actual)
                    batch_memory = random.sample(memory_buffer, mem_batch_size)

                    X_mem = [item[0] for item in batch_memory]
                    y_decoder_mem = [item[0] for item in batch_memory] 
                    y_ae_targets_mem = [item[1] for item in batch_memory]
                    y_targets_mem = [item[1] for item in batch_memory]

                    # Convert lists to tensors and move to device
                    X_mem = torch.stack(X_mem).to(device)
                    y_decoder_mem = torch.stack(y_decoder_mem).to(device)
                    y_ae_targets_mem = torch.stack(y_ae_targets_mem).to(device)
                    y_targets_mem = torch.stack(y_targets_mem).to(device)

                    # Combine current batch with memory samples
                    X_combined = torch.cat([X_batch, X_mem], dim=0)
                    y_decoder_combined = torch.cat([y_decoder_batch, y_decoder_mem], dim=0)
                    y_ae_targets_combined = torch.cat([y_ae_targets_batch, y_ae_targets_mem], dim=0)
                    y_targets_combined = torch.cat([y_targets_batch, y_targets_mem], dim=0)
                else:
                    X_combined = X_batch
                    y_decoder_combined = y_decoder_batch
                    y_ae_targets_combined = y_ae_targets_batch
                    y_targets_combined = y_targets_batch

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                out_encoder, decoder_output, out_ae_output, out_output = model(X_combined)

                # Compute losses
                loss_decoder = criterion_decoder(decoder_output, y_decoder_combined)
                loss_ae_targets = criterion_ae_targets(out_ae_output, y_ae_targets_combined)
                loss_targets = criterion_targets(out_output, y_targets_combined)
                loss_enc_targets = criterion_enc_targets(out_encoder, y_targets_combined)
                loss = loss_decoder + loss_ae_targets + loss_targets + loss_enc_targets

                # Backward pass and SGD update
                loss.backward()
                optimizer.step()

            # Save θ_W_ik
            theta_W_ik = deepcopy(model.state_dict())

            # Within-batch Reptile meta-update
            for param_name in theta_W_i0:
                theta_W_i0[param_name] = theta_W_i0[param_name] + beta * (theta_W_ik[param_name] - theta_W_i0[param_name])
            model.load_state_dict(theta_W_i0)

            # Update θ_A_s for across-batch meta-update
            theta_A_s = model.state_dict()

        # Across-batch Reptile meta-update
        for param_name in theta_A0:
            theta_A0[param_name] = theta_A0[param_name] + gamma * (theta_A_s[param_name] - theta_A0[param_name])
        model.load_state_dict(theta_A0)

        # Update memory buffer 
        memory_buffer, age = update_memory_buffer(memory_buffer, buffer_size, age, X_batch.cpu(), y_targets_batch.cpu())

        # Accumulate losses
        total_loss += loss.item()
        running_loss_decoder += loss_decoder.item()
        running_loss_ae_targets += loss_ae_targets.item()
        running_loss_targets += loss_targets.item()
        running_loss_enc_pred += loss_enc_targets.item()

    # Compute average losses for the era
    num_batches = len(dataloader)
    average_loss = total_loss / num_batches
    running_loss_decoder = running_loss_decoder / num_batches
    running_loss_ae_targets = running_loss_ae_targets / num_batches
    running_loss_targets = running_loss_targets / num_batches
    running_loss_enc_pred = running_loss_enc_pred / num_batches

    # Log the losses to the text file
    with open('mer1_training_losses.txt', 'a') as log_file:
        log_file.write(f'{era}\t{average_loss:.6f}\t{running_loss_decoder:.6f}\t{running_loss_ae_targets:.6f}\t{running_loss_targets:.6f}\t{running_loss_enc_pred:.6f}\n')

    # Optionally, print the losses
    print(f"Era {era} | Total Loss: {average_loss:.6f} | Decoder Loss: {running_loss_decoder:.6f} | AE_target Loss: {running_loss_ae_targets:.6f} | Target Loss: {running_loss_targets:.6f} | Encoder Pred Loss: {running_loss_enc_pred:.6f}")

    # Save the model after every 100 eras
    if era_idx % 100 == 0:
        model_filename = f'model_era_{era_idx}.pt'
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved to {model_filename}')


model_filename = f'last_model.pt'
torch.save(model.state_dict(), model_filename)
