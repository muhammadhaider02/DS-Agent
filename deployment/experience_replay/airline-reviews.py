import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer, DebertaModel, DebertaTokenizer
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return rmse


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


class DistilBERTRegressor(nn.Module):
    def __init__(self):
        super(DistilBERTRegressor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = nn.Linear(self.distilbert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output).squeeze(-1)


class RoBERTaRegressor(nn.Module):
    def __init__(self):
        super(RoBERTaRegressor, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output).squeeze(-1)


class DeBERTaRegressor(nn.Module):
    def __init__(self):
        super(DeBERTaRegressor, self).__init__()
        # Use safetensors to avoid torch.load vulnerability
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-large', use_safetensors=True)
        
        # GeM pooling parameter - initialized to 3.0 as requested
        self.p = nn.Parameter(torch.tensor(3.0))
        
        # Enhanced regression head with additional linear layer
        self.regressor = nn.Sequential(
            nn.Linear(self.deberta.config.hidden_size, self.deberta.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.deberta.config.hidden_size, 1)
        )
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Get the last hidden states (sequence output)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply GeM pooling
        # Expand attention mask to match hidden_size dimension
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        
        # Apply GeM formula: (1/N * Σ x_i^p)^(1/p)
        # First, apply absolute value and power p (with small epsilon to avoid NaN)
        eps = 1e-6
        powered = torch.abs(sequence_output) + eps
        powered = torch.pow(powered, self.p)
        
        # Apply attention mask (zero out padding tokens)
        powered = powered * attention_mask_expanded
        
        # Sum over sequence dimension
        sum_powered = torch.sum(powered, dim=1)  # [batch_size, hidden_size]
        
        # Count non-padding tokens
        sum_mask = torch.sum(attention_mask_expanded, dim=1)  # [batch_size, 1]
        
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # Compute mean of powered values
        mean_powered = sum_powered / sum_mask
        
        # Apply inverse power (1/p)
        gem_output = torch.pow(mean_powered, 1.0/self.p)
        
        # Apply dropout
        gem_output = self.dropout(gem_output)
        
        # Pass through enhanced regression head
        return self.regressor(gem_output).squeeze(-1)


def train_single_model(model_name, X_train, y_train, X_valid, y_valid):
    # Load tokenizer and model based on model type
    if model_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBERTRegressor().to(device)
        batch_size = 8
        max_length = 256
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RoBERTaRegressor().to(device)
        batch_size = 8
        max_length = 256
    elif model_name == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
        model = DeBERTaRegressor().to(device)
        # Reduce batch size and sequence length for DeBERTa-Large to save memory
        batch_size = 4
        max_length = 192
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, max_length=max_length)
    valid_dataset = ReviewDataset(X_valid, y_valid, tokenizer, max_length=max_length)

    # Create data loaders with adjusted batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Calculate total training steps for scheduler
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    # Create linear scheduler with warmup
    scheduler = lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0/3.0,  # Start at 1/3 of max LR
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Increase gradient accumulation steps for DeBERTa-Large
    accumulation_steps = 4 if model_name == 'deberta' else 2

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(valid_loader)
        val_rmse = compute_metrics_for_regression(all_labels, all_preds)

        print(f"{model_name.upper()} - Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)
    return model, tokenizer


def predict_single_model(model, tokenizer, X):
    model.eval()
    # Use smaller batch size for prediction if it's DeBERTa-Large
    batch_size = 4 if isinstance(model, DeBERTaRegressor) else 8
    dataset = ReviewDataset(X, None, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.cpu().numpy()

            # Clip predictions to reasonable range (1-10 for ratings)
            predictions = np.clip(predictions, 1, 10)
            all_predictions.extend(predictions)

    return np.array(all_predictions)


def find_optimal_weights(valid_predictions, y_valid):
    """Find optimal weights for ensemble averaging using grid search."""
    best_rmse = float('inf')
    best_weights = (0.33, 0.33, 0.34)  # Default equal weights
    
    # Generate more granular weight combinations
    weight_values = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 0.9]
    
    for w1 in weight_values:
        for w2 in weight_values:
            w3 = 1.0 - w1 - w2
            # Allow small tolerance for floating point errors
            if w3 >= 0.05 and w3 <= 0.95:
                # Weighted average
                weighted_pred = (w1 * valid_predictions[0] + 
                                w2 * valid_predictions[1] + 
                                w3 * valid_predictions[2])
                weighted_pred = np.clip(weighted_pred, 1, 10)
                
                # Compute RMSE
                rmse = compute_metrics_for_regression(y_valid, weighted_pred)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = (w1, w2, w3)
    
    print(f"\nOptimal weights found: DistilBERT={best_weights[0]:.2f}, "
          f"RoBERTa={best_weights[1]:.2f}, DeBERTa={best_weights[2]:.2f}")
    print(f"Validation RMSE with optimal weights: {best_rmse:.4f}")
    
    return best_weights


def train_ensemble(X_train, y_train, X_valid, y_valid):
    print("Training DistilBERT model...")
    distilbert_model, distilbert_tokenizer = train_single_model(
        'distilbert', X_train, y_train, X_valid, y_valid
    )
    
    print("\nTraining RoBERTa model...")
    roberta_model, roberta_tokenizer = train_single_model(
        'roberta', X_train, y_train, X_valid, y_valid
    )
    
    print("\nTraining DeBERTa model...")
    deberta_model, deberta_tokenizer = train_single_model(
        'deberta', X_train, y_train, X_valid, y_valid
    )
    
    # Get validation predictions from each model
    print("\nGenerating validation predictions for weight optimization...")
    distilbert_preds = predict_single_model(distilbert_model, distilbert_tokenizer, X_valid)
    roberta_preds = predict_single_model(roberta_model, roberta_tokenizer, X_valid)
    deberta_preds = predict_single_model(deberta_model, deberta_tokenizer, X_valid)
    
    # Find optimal weights using grid search
    valid_predictions = [distilbert_preds, roberta_preds, deberta_preds]
    optimal_weights = find_optimal_weights(valid_predictions, y_valid)
    
    return {
        'distilbert': (distilbert_model, distilbert_tokenizer),
        'roberta': (roberta_model, roberta_tokenizer),
        'deberta': (deberta_model, deberta_tokenizer),
        'weights': optimal_weights
    }


def predict_ensemble(ensemble, X):
    all_predictions = []
    
    # Get predictions from each model
    for model_name in ['distilbert', 'roberta', 'deberta']:
        model, tokenizer = ensemble[model_name]
        print(f"Generating predictions with {model_name}...")
        predictions = predict_single_model(model, tokenizer, X)
        all_predictions.append(predictions)
    
    # Get optimal weights
    weights = ensemble['weights']
    
    # Weighted average using optimal weights
    ensemble_predictions = (weights[0] * all_predictions[0] + 
                           weights[1] * all_predictions[1] + 
                           weights[2] * all_predictions[2])
    
    # Clip final predictions
    ensemble_predictions = np.clip(ensemble_predictions, 1, 10)
    
    return ensemble_predictions


if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['OverallRating'])

    # Process data and store into numpy arrays.
    X = list(data_df.ReviewBody.to_numpy())
    y = data_df.OverallRating.to_numpy().astype(np.float32)

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # Train ensemble of models
    ensemble = train_ensemble(X_train, y_train, X_valid, y_valid)

    # Evaluate the ensemble on the valid set
    y_valid_pred = predict_ensemble(ensemble, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"\nFinal Ensemble RMSE on validation set: {rmse:.4f}")

    # Submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['OverallRating'])
    X_submission = list(submission_df.ReviewBody.to_numpy())
    y_submission = predict_ensemble(ensemble, X_submission)
    submit_predictions_for_test_set(y_submission)