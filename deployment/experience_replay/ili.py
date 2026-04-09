import torch
import numpy as np
import random
import os

from submission import submit_predictions_for_test_set
from dataset import get_dataset

# get_dataset function returns X and y with:
# X with shape of (NUM, INPUT_SEQ_LEN, INPUT_DIM)
# y with shape of (NUM, PRED_SEQ_LEN, PRED_DIM)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_SEQ_LEN = 36
INPUT_DIM = 7
PRED_SEQ_LEN = 24
PRED_DIM = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_for_time_series_forecasting(y_test, y_test_pred):
    y_test = y_test.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    y_test_pred = y_test_pred.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    mae = np.mean(np.abs(y_test - y_test_pred))
    mse = np.mean((y_test - y_test_pred)**2)
    return mae, mse


class ResidualBiGRU(torch.nn.Module):
    def __init__(self):
        super(ResidualBiGRU, self).__init__()
        
        # Bidirectional GRU layer
        self.gru = torch.nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear projection for residual connection (matching GRU output dimension)
        self.residual_proj = torch.nn.Linear(INPUT_DIM, 128)  # 64*2 due to bidirectional
        
        # Dropout layer with increased rate
        self.dropout1 = torch.nn.Dropout(0.5)
        
        # Linear layers for output projection
        # Input to linear1 should be (batch_size, INPUT_SEQ_LEN * 128) after flattening
        self.linear1 = torch.nn.Linear(INPUT_SEQ_LEN * 128, 256)
        self.relu1 = torch.nn.ReLU()
        # Additional dropout layer after first linear layer
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(256, PRED_SEQ_LEN * PRED_DIM)
        
    def forward(self, x):
        # x shape: (batch_size, INPUT_SEQ_LEN, INPUT_DIM)
        
        # GRU layer
        gru_out, _ = self.gru(x)  # shape: (batch, seq_len, hidden_size*2=128)
        
        # Project input for residual connection
        residual = self.residual_proj(x)  # shape: (batch, seq_len, 128)
        
        # Residual connection: add projected input to GRU output
        gru_out = gru_out + residual
        
        # Apply first dropout
        gru_out = self.dropout1(gru_out)
        
        # Flatten the sequence dimension for processing through linear layers
        batch_size = gru_out.size(0)
        gru_flat = gru_out.reshape(batch_size, -1)  # shape: (batch, INPUT_SEQ_LEN * 128)
        
        # Pass through feed-forward layers
        x = self.linear1(gru_flat)
        x = self.relu1(x)
        # Apply second dropout
        x = self.dropout2(x)
        x = self.linear2(x)
        
        # Reshape to output sequence
        x = x.view(-1, PRED_SEQ_LEN, PRED_DIM)
        
        return x


def train_model(X_train, y_train, X_valid, y_valid):
    # Compute normalization statistics from training data
    # Reshape to (num_samples * time_steps, features)
    X_train_reshaped = X_train.reshape(-1, INPUT_DIM)
    y_train_reshaped = y_train.reshape(-1, PRED_DIM)
    
    # Compute mean and std for each feature
    X_mean = np.mean(X_train_reshaped, axis=0)
    X_std = np.std(X_train_reshaped, axis=0)
    y_mean = np.mean(y_train_reshaped, axis=0)
    y_std = np.std(y_train_reshaped, axis=0)
    
    # Avoid division by zero
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    
    # Normalize training data
    X_train_norm = (X_train - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    
    # Normalize validation data
    X_valid_norm = (X_valid - X_mean) / X_std
    y_valid_norm = (y_valid - y_mean) / y_std
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train_norm).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid_norm).to(device)
    y_valid_tensor = torch.FloatTensor(y_valid_norm).to(device)
    
    # Create model
    model = ResidualBiGRU().to(device)
    
    # Loss function - Huber loss (combines MSE and MAE benefits)
    criterion = torch.nn.HuberLoss(delta=1.0)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    # Training parameters
    batch_size = 8
    max_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    # For checkpoint averaging
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_checkpoints = []
    
    # Training loop
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle training data
        indices = torch.randperm(X_train_tensor.size(0))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]
        
        # Mini-batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = epoch_loss / len(X_train_tensor)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_valid_tensor)
            val_loss = criterion(val_outputs, y_valid_tensor).item()
        
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save checkpoint for averaging (last 10 epochs or all if fewer than 10)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
        }, checkpoint_path)
        
        saved_checkpoints.append(checkpoint_path)
        
        # Keep only the last 10 checkpoints
        if len(saved_checkpoints) > 10:
            # Remove the oldest checkpoint
            os.remove(saved_checkpoints[0])
            saved_checkpoints.pop(0)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Store normalization parameters and checkpoint paths in the model
    model.X_mean = torch.FloatTensor(X_mean).to(device)
    model.X_std = torch.FloatTensor(X_std).to(device)
    model.y_mean = torch.FloatTensor(y_mean).to(device)
    model.y_std = torch.FloatTensor(y_std).to(device)
    model.saved_checkpoints = saved_checkpoints
    
    return model

def predict(model, X):
    # If we have saved checkpoints, use checkpoint averaging
    if hasattr(model, 'saved_checkpoints') and len(model.saved_checkpoints) > 0:
        all_preds = []
        
        # Load each checkpoint and make predictions
        for checkpoint_path in model.saved_checkpoints:
            # Create a new model instance
            checkpoint_model = ResidualBiGRU().to(device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            checkpoint_model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_model.eval()
            
            # Get normalization parameters from checkpoint
            X_mean = checkpoint['X_mean']
            X_std = checkpoint['X_std']
            y_mean = checkpoint['y_mean']
            y_std = checkpoint['y_std']
            
            # Normalize input using checkpoint's normalization parameters
            X_norm = (X - X_mean) / X_std
            X_tensor = torch.FloatTensor(X_norm).to(device)
            
            # Make predictions
            with torch.no_grad():
                preds_norm = checkpoint_model(X_tensor).cpu().numpy()
            
            # Denormalize predictions using checkpoint's normalization parameters
            preds = preds_norm * y_std + y_mean
            all_preds.append(preds)
        
        # Average predictions from all checkpoints
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds
    else:
        # Fallback to single model prediction if no checkpoints
        model.eval()
        
        # Normalize input
        X_norm = (X - model.X_mean.cpu().numpy()) / model.X_std.cpu().numpy()
        X_tensor = torch.FloatTensor(X_norm).to(device)
        
        with torch.no_grad():
            preds_norm = model(X_tensor).cpu().numpy()
        
        # Denormalize predictions
        preds = preds_norm * model.y_std.cpu().numpy() + model.y_mean.cpu().numpy()
        
        return preds


if __name__ == '__main__':
    # Load training set
    X_train, y_train = get_dataset(flag='train')
    # Load validation set
    X_valid, y_valid = get_dataset(flag='val')

    # define and train the model
    # should fill out the train_model function
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_time_series_forecasting and print the results
    # should fill out the predict function
    y_valid_pred = predict(model, X_valid)
    mae, mse = compute_metrics_for_time_series_forecasting(y_valid, y_valid_pred)
    print(f"Final MSE on validation set: {mse}, Final MAE on validation set: {mae}.")

    # Submit predictions on the test set
    X_test, y_test = get_dataset(flag='test')
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test, y_test_pred)