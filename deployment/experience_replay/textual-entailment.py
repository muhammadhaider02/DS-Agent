import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_for_classification(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred) 
    return acc


class TextualEntailmentDataset(Dataset):
    def __init__(self, texts1, texts2, labels=None, tokenizer=None, max_length=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        
        encoding = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


class GeMPooling(nn.Module):
    def __init__(self, hidden_size, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps
        
    def forward(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        # Expand attention mask for broadcasting
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # Apply mask to embeddings (zero out padding tokens)
        masked_embeddings = token_embeddings * attention_mask_expanded
        
        # Compute GeM pooling
        # For numerical stability, we compute (|x|^p + eps)^(1/p)
        p = torch.clamp(self.p, min=1e-6)  # Ensure p > 0
        
        # Compute |x|^p
        abs_embeddings = torch.abs(masked_embeddings)
        powered = torch.pow(abs_embeddings + self.eps, p)
        
        # Sum over sequence dimension (excluding padding)
        sum_powered = torch.sum(powered, dim=1)
        
        # Count non-padding tokens for each sequence
        valid_counts = torch.sum(attention_mask_expanded, dim=1)
        
        # Compute mean of powered values
        mean_powered = sum_powered / torch.clamp(valid_counts, min=1)
        
        # Compute (mean)^(1/p)
        pooled_output = torch.pow(mean_powered, 1.0/p)
        
        return pooled_output


class RobertaForTextualEntailmentGeM(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('roberta-large', use_safetensors=True)
        self.gem_pooling = GeMPooling(self.roberta.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use GeM pooling instead of [CLS]
        token_embeddings = outputs.last_hidden_state
        pooled_output = self.gem_pooling(token_embeddings, attention_mask)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DebertaLargeForTextualEntailmentMean(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-large', use_safetensors=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling across all token representations, excluding padding tokens
        token_embeddings = outputs.last_hidden_state
        
        # Expand attention mask dimensions for broadcasting
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # Sum the embeddings, excluding padding tokens
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        
        # Count the number of non-padding tokens for each sequence
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        
        # Compute mean pooling
        pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class EnsembleModel:
    def __init__(self, model1, model2, tokenizer, weight1=0.5, averaging_method='logits'):
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokenizer
        self.weight1 = weight1
        self.weight2 = 1.0 - weight1
        self.averaging_method = averaging_method
        
        # Ensure models are in eval mode and on correct device
        self.model1.eval()
        self.model2.eval()
        self.model1.to(device)
        self.model2.to(device)
        
    def predict(self, X):
        self.model1.eval()
        self.model2.eval()
        predictions = []
        
        # Create dataset
        dataset = TextualEntailmentDataset(
            X[:, 0], X[:, 1], None, self.tokenizer, max_length=128
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Get logits from both models
                logits1 = self.model1(input_ids, attention_mask)
                logits2 = self.model2(input_ids, attention_mask)
                
                if self.averaging_method == 'logits':
                    # Weighted averaging of logits
                    weighted_logits = self.weight1 * logits1 + self.weight2 * logits2
                    preds = torch.argmax(weighted_logits, dim=1)
                elif self.averaging_method == 'probabilities':
                    # Convert logits to probabilities using softmax
                    probs1 = torch.softmax(logits1, dim=1)
                    probs2 = torch.softmax(logits2, dim=1)
                    # Weighted averaging of probabilities
                    weighted_probs = self.weight1 * probs1 + self.weight2 * probs2
                    preds = torch.argmax(weighted_probs, dim=1)
                else:
                    raise ValueError(f"Unknown averaging method: {self.averaging_method}")
                
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)


def train_single_model(model_name, X_train, y_train, X_valid, y_valid):
    if model_name == 'roberta_gem':
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        model = RobertaForTextualEntailmentGeM(num_labels=3)
    elif model_name == 'deberta_large':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large')
        model = DebertaLargeForTextualEntailmentMean(num_labels=3)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    
    # Create datasets
    train_dataset = TextualEntailmentDataset(
        X_train[:, 0], X_train[:, 1], y_train, tokenizer, max_length=128
    )
    valid_dataset = TextualEntailmentDataset(
        X_valid[:, 0], X_valid[:, 1], y_valid, tokenizer, max_length=128
    )
    
    # Create data loaders with small batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5
    accumulation_steps = 2  # Effective batch size = 8 * 2 = 16
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        
        valid_acc = accuracy_score(valid_labels, valid_preds)
        print(f"{model_name.upper()} - Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Acc: {valid_acc:.4f}")
    
    return model, tokenizer


def train_ensemble(X_train, y_train, X_valid, y_valid):
    print("Training RoBERTa-large model with GeM pooling...")
    roberta_model, roberta_tokenizer = train_single_model('roberta_gem', X_train, y_train, X_valid, y_valid)
    
    print("\nTraining DeBERTa-Large model with mean pooling...")
    deberta_model, deberta_tokenizer = train_single_model('deberta_large', X_train, y_train, X_valid, y_valid)
    
    return roberta_model, deberta_model, roberta_tokenizer


def grid_search_ensemble(roberta_model, deberta_model, tokenizer, X_valid, y_valid):
    print("\nPerforming grid search for ensemble parameters...")
    
    best_acc = 0.0
    best_weight = 0.5
    best_method = 'logits'
    
    weights = [i/10.0 for i in range(0, 11)]  # 0.0 to 1.0 in steps of 0.1
    averaging_methods = ['logits', 'probabilities']
    
    for weight in weights:
        for method in averaging_methods:
            ensemble = EnsembleModel(roberta_model, deberta_model, tokenizer, 
                                    weight1=weight, averaging_method=method)
            y_valid_pred = ensemble.predict(X_valid)
            acc = accuracy_score(y_valid, y_valid_pred)
            
            print(f"Weight: {weight:.1f}, Method: {method}, Validation Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_weight = weight
                best_method = method
    
    print(f"\nBest ensemble configuration:")
    print(f"  Weight for RoBERTa: {best_weight:.1f}")
    print(f"  Averaging method: {best_method}")
    print(f"  Validation Accuracy: {best_acc:.4f}")
    
    # Create final ensemble with best parameters
    best_ensemble = EnsembleModel(roberta_model, deberta_model, tokenizer,
                                 weight1=best_weight, averaging_method=best_method)
    
    return best_ensemble, best_weight, best_method, best_acc


def predict_ensemble(ensemble_model, X):
    return ensemble_model.predict(X)


if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    # Process data and store into numpy arrays.
    X = data_df[["text1", "text2"]].to_numpy()
    y = data_df.label.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # Train individual models
    roberta_model, deberta_model, tokenizer = train_ensemble(X_train, y_train, X_valid, y_valid)

    # Perform grid search to find best ensemble parameters
    best_ensemble, best_weight, best_method, best_acc = grid_search_ensemble(
        roberta_model, deberta_model, tokenizer, X_valid, y_valid
    )

    # Submit predictions for the test set using best ensemble
    submission_df = pd.read_csv('test.csv')
    X_submission = submission_df[["text1", "text2"]].to_numpy()
    y_submission = predict_ensemble(best_ensemble, X_submission)
    submit_predictions_for_test_set(y_submission)