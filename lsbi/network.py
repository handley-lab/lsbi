"""Simple binary classifiers to perform model comparison."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class BinaryClassifierBase(nn.Module):
    """Base model for binary classification. Following 2305.11241.

    A simple binary classifier:
        - 5 hidden layers:
            - Layer 1 with initial_dim units
            - Layers 2-4 with internal_dim units
        - Leaky ReLU activation function
        - Batch normalization
        - Output layer with 1 unit linear classifier unit
        - Adam optimizer with default learning rate 0.001
        - Exponential learning rate decay with default decay rate 0.95

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    internal_dim : int, optional (default=16)
        Dimension of the internal layers of the network.
    initial_dim : int, optional (default=130)
        Dimension of the first layer of the network.
    """

    def __init__(self, input_dim, internal_dim=16, initial_dim=130):
        super(BinaryClassifierBase, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, initial_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(initial_dim),
            nn.Linear(initial_dim, internal_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.LeakyReLU(),
            nn.Linear(internal_dim, 1),
        )

    def forward(self, x):
        """Forward pass through the network, logit output."""
        return self.model(x)

    def loss(self, x, y):
        """Loss function for the network."""
        raise NotImplementedError

    def predict(self, x):
        """Predict the Bayes Factor."""
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        """Fit classifier on input features X to predict labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target values.
        num_epochs : int, optional (default=10)
            Number of epochs to train the network.
        batch_size : int, optional (default=128)
            Batch size for training.
        decay_rate : float, optional (default=0.95)
            Decay rate for the learning rate scheduler.
        lr : float, optional (default=0.001)
            Learning rate for the optimizer.
        device : str, optional (default="cpu")
            Device to use for training.
        """
        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        decay_rate = kwargs.get("decay_rate", 0.95)
        lr = kwargs.get("lr", 0.001)
        device = torch.device(kwargs.get("device", "cpu"))

        print("Using device: ", device)

        # Convert labels to torch tensor
        X = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)

        # Create a DataLoader for batch training
        dataset = torch.utils.data.TensorDataset(X, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Define the loss function and optimizer
        criterion = self.loss
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Create the scheduler and pass in the optimizer and decay rate
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)

        # Create a DataLoader for batch training
        self.to(device=device, dtype=torch.float32)

        for epoch in range(num_epochs):
            epoch_loss = []
            for i, (inputs, targets) in enumerate(dataloader):
                # Clear gradients
                optimizer.zero_grad()
                inputs = inputs.to(device)
                # Forward pass
                loss = criterion(inputs, targets)
                epoch_loss.append(loss.item())
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Print loss for every epoch
            scheduler.step()
            mean_loss = torch.mean(torch.tensor(epoch_loss)).item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss}")

        # once training is done, set the model to eval(), ensures batchnorm
        # and dropout are not used during inference
        self.model.eval()


class BinaryClassifier(BinaryClassifierBase):
    """
    Extends the BinaryClassifierBase to use a BCE loss function.

    Furnishes with a direction prediction of the Bayes Factor.
    """

    def loss(self, x, y):
        """Binary cross entropy loss function for the network."""
        y_ = self.forward(x)
        return nn.BCEWithLogitsLoss()(y_, y)

    def predict(self, x):
        """Predict the log Bayes Factor.

        log K = lnP(Class 1) - lnP(Class 0)
        """
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = nn.Sigmoid()(pred)
        return (torch.log(pred) - torch.log(1 - pred)).detach().numpy()


class BinaryClassifierLPop(BinaryClassifierBase):
    """
    Extends the BinaryClassifierBase to use a LPop Exponential loss.

    Furnishes with a direction prediction of the Bayes Factor.

    Parameters
    ----------
    alpha : float, optional (default=2.0)
        Scale factor for the exponent transform.
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop("alpha", 2.0)
        super(BinaryClassifierLPop, self).__init__(*args, **kwargs)

    def lpop(self, x):
        """Leaky parity odd power transform."""
        return x + x * torch.pow(torch.abs(x), self.alpha - 1.0)

    def loss(self, x, y):
        """Lpop Loss function for the network."""
        x = self.forward(x)
        return torch.exp(
            torch.logsumexp((0.5 - y) * self.lpop(x), dim=0)
            - torch.log(torch.tensor(x.shape[0], dtype=torch.float64))
        ).squeeze()

    def predict(self, x):
        """Predict the log Bayes Factor.

        log K = lnP(Class 1) - lnP(Class 0)
        """
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = self.lpop(pred)
        return pred.detach().numpy()
