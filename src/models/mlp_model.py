import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dims=[100, 50], epochs=30, lr=0.001, batch_size=32, verbose=False):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(self.hidden_dims)):
            layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.hidden_dims[-1], 1))  
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            if self.verbose and (epoch % 10 == 0):
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        return self

    def predict(self, X):
        self.model.eval()
        X = check_array(X)
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            preds = self.model(X).cpu().numpy().flatten()
        return preds