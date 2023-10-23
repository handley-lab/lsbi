"""Simple binary classifiers to perform model comparison."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class BinaryClassifierBase(nn.Module):
    """Base model for binary classification. Following 2305.11241."""

    def __init__(self, input_dim, internal_dim=16, initial_dim=130):
        super(BinaryClassifierBase, self).__init__()

        self.dense = nn.Linear(input_dim, initial_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(initial_dim)
        self.dense_1 = nn.Linear(initial_dim, internal_dim)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.batch_norm_1 = nn.BatchNorm1d(internal_dim)
        self.dense_2 = nn.Linear(internal_dim, internal_dim)
        self.leaky_relu_2 = nn.LeakyReLU()
        self.batch_norm_2 = nn.BatchNorm1d(internal_dim)
        self.dense_3 = nn.Linear(internal_dim, internal_dim)
        self.leaky_relu_3 = nn.LeakyReLU()
        self.batch_norm_3 = nn.BatchNorm1d(internal_dim)
        self.dense_4 = nn.Linear(internal_dim, internal_dim)
        self.leaky_relu_4 = nn.LeakyReLU()
        self.dense_5 = nn.Linear(internal_dim, 1)

    def forward(self, x):
        """Forward pass through the network, logit output."""
        x = self.dense(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        x = self.batch_norm_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.batch_norm_2(x)
        x = self.dense_3(x)
        x = self.leaky_relu_3(x)
        x = self.batch_norm_3(x)
        x = self.dense_4(x)
        x = self.leaky_relu_4(x)
        x = self.dense_5(x)
        return x

    def loss(self, x):
        """Loss function for the network."""
        raise NotImplementedError

    def predict(self, x):
        """Predict the Bayes Factor."""
        raise NotImplementedError


class BinaryClassifier(BinaryClassifierBase):
    """
    Extends the BinaryClassifierBase to use a BCE loss function.

    Furnishes with a direction prediction of the Bayes Factor.
    """

    def loss(self, x, target):
        """Binary cross entropy loss function for the network."""
        x = self.forward(x)
        return nn.BCEWithLogitsLoss()(x, target)

    def predict(self, x):
        """Predict the log Bayes Factor.

        log K = lnP(Class 1) - lnP(Class 0)
        """
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = nn.Sigmoid()(pred)
        return (torch.log(pred) - torch.log(1-pred)).detach().numpy()


class BinaryClassifierLPop(BinaryClassifierBase):
    """
    Extends the BinaryClassifierBase to use a LPop Exponential loss.

    Furnishes with a direction prediction of the Bayes Factor.
    """

    def lpop(self, x, alpha=2.0):
        """Leaky parity odd power transform."""
        return x + x * torch.pow(torch.abs(x), alpha - 1.0)

    def loss(self, x, target, alpha=2.0):
        """Lpop Loss function for the network."""
        x = self.forward(x)
        return torch.exp(
            torch.logsumexp((0.5 - target) * self.lpop(x, alpha=alpha), dim=0)
            - torch.log(torch.tensor(x.shape[0], dtype=torch.float64))
        ).squeeze()

    def predict(self, x, alpha=2.0):
        """Predict the log Bayes Factor.

        log K = lnP(Class 1) - lnP(Class 0)
        """
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = self.lpop(pred, alpha=alpha)
        return pred.detach().numpy()


def train(
    model,
    data,
    labels,
    num_epochs=10,
    batch_size=128,
    decay_rate=0.95,
    lr=0.001,
):
    """Train the binary classifier."""
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")

    # for now just restrict to cpu
    device = torch.device("cpu")

    # device = torch.device("mps")
    print("Using device: ", device)

    # Convert labels to torch tensor
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    labels = labels.unsqueeze(1)

    batch_size = batch_size
    labels = labels.to(device)
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Define the loss function and optimizer
    criterion = model.loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create the scheduler and pass in the optimizer and decay rate
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    # Train the binary classifier
    num_epochs = num_epochs

    # Create a DataLoader for batch training
    model = model.to(torch.float32)
    # data = data.to(device)
    model = model.to(device)

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

    # Evaluate the model
    model.to("cpu")
    model.batch_norm.eval()
    model.batch_norm_1.eval()
    model.batch_norm_2.eval()
    model.batch_norm_3.eval()
    return model
