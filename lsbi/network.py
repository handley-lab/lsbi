import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class BinaryClassifierBase(nn.Module):
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
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class BinaryClassifier(BinaryClassifierBase):
    def loss(self, x, target):
        x=self.forward(x)
        return nn.BCEWithLogitsLoss()(x, target)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = nn.Sigmoid()(pred)
        return (torch.log(1 - pred) - torch.log(pred)).detach().numpy()


class BinaryClassifierLPop(BinaryClassifierBase):
    def lpop(self, x, alpha=2.0):
        """implements leaky parity odd power transform"""
        return x + x * torch.pow(torch.abs(x), alpha - 1.0)

    def loss(self, x, target, alpha=2.0):
        x=self.forward(x)
        return torch.exp(
            torch.logsumexp((0.5 - target) * self.lpop(x, alpha=alpha), dim=0)
            - torch.log(torch.tensor(x.shape[0], dtype=torch.float64))
        ).squeeze()

    def predict(self, x, alpha=2.0):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        pred = self.lpop(pred,alpha=alpha)
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
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
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

    best_loss = float("inf")
    best_weights = None
    patience = 5
    counter = 0
    for epoch in range(num_epochs):
        epoch_loss = []
        for i, (inputs, targets) in enumerate(dataloader):
            # Clear gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            # Forward pass
            # outputs = model(inputs)
            loss = criterion(inputs, targets)
            epoch_loss.append(loss.item())
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print loss for every epoch
        scheduler.step()
        mean_loss = torch.mean(torch.tensor(epoch_loss)).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"No improvement for {patience} epochs. Rolling back to best epoch."
                )
                model.load_state_dict(best_weights)
                break
    # Evaluate the model
    model.to("cpu")
    model.batch_norm.eval()
    model.batch_norm_1.eval()
    model.batch_norm_2.eval()
    model.batch_norm_3.eval()
    return model
