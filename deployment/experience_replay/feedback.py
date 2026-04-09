import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, AutoConfig

from submission import submit_predictions_for_test_set

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        rmse = np.sqrt(mean_squared_error(targets_task, pred_task))
        metrics[f"rmse_{task}"] = rmse
    rmse = np.mean(list(metrics.values()))
    return rmse


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
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
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
            
        return item


class DeBERTaRegressor(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=6):
        super(DeBERTaRegressor, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        # Load model with safetensors to avoid torch.load vulnerability check
        self.deberta = AutoModel.from_pretrained(
            model_name, 
            config=config,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
            use_safetensors=True  # Use safetensors format
        )
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.deberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.float()
        logits = self.regressor(pooled_output)
        return logits


def train_model(X_train, y_train, X_valid, y_valid):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DeBERTaRegressor()
    model.to(device)
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    valid_dataset = TextDataset(X_valid, y_valid, tokenizer)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    num_epochs = 3
    best_val_rmse = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                valid_preds.extend(outputs.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        
        valid_rmse = compute_metrics_for_regression(valid_labels, valid_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation RMSE: {valid_rmse:.4f}')
        print('-' * 50)
        
        # Save best model
        if valid_rmse < best_val_rmse:
            best_val_rmse = valid_rmse
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with RMSE: {best_val_rmse:.4f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, tokenizer


def predict(model, tokenizer, X):
    model.eval()
    
    # Create dataset and dataloader
    dataset = TextDataset(X, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


if __name__ == '__main__':
    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)

    # Process data and store into numpy arrays.
    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X))])

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model, tokenizer = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, tokenizer, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final MCRMSE on validation set: ", rmse)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv',  header=0, names=['text_id', 'full_text'], index_col='text_id')
    X_submission = list(submission_df.full_text.to_numpy())
    y_submission = predict(model, tokenizer, X_submission)
    submit_predictions_for_test_set(y_submission)