from lsbi.network import (
    BinaryClassifierBase,
    BinaryClassifier,
    BinaryClassifierLPop,
    train,
)
import torch
import numpy as np
import pytest


@pytest.mark.parametrize("input_dim", [1, 100])
@pytest.mark.parametrize("internal_dim", [16, 32])
@pytest.mark.parametrize("initial_dim", [130, 256])
class TestClassifierBase:
    CLS = BinaryClassifierBase

    @pytest.fixture
    def model(self, input_dim, internal_dim, initial_dim):
        return self.CLS(input_dim, internal_dim, initial_dim)

    @pytest.fixture
    def x(self, input_dim):
        return torch.tensor(np.random.rand(10, input_dim), dtype=torch.float32)

    @pytest.fixture
    def y(self):
        return torch.tensor(
            np.random.randint(0, 2, size=(10, 1)), dtype=torch.float32
        )

    def test_init(self, model):
        assert isinstance(model, BinaryClassifierBase)

    def test_forward(self, model, x):
        y = model.forward(x)
        assert y.shape == (10, 1)

    def test_loss(self, model, x):
        with pytest.raises(NotImplementedError):
            model.loss(x)

    def test_predict(self, model, x):
        with pytest.raises(NotImplementedError):
            model.predict(x)


class TestClassifier(TestClassifierBase):
    CLS = BinaryClassifier

    def test_loss(self, model, x, y):
        loss = model.loss(x, y)
        assert loss.detach().numpy().shape == ()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_predict(self, model, x):
        y = model.predict(x)
        assert y.shape == (10, 1)
        assert isinstance(y, np.ndarray)


class TestClassifierLPop(TestClassifierBase):
    CLS = BinaryClassifierLPop

    def test_loss(self, model, x, y):
        loss = model.loss(x, y)
        assert loss.detach().numpy().shape == ()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_predict(self, model, x):
        y = model.predict(x)
        assert y.shape == (10, 1)
        assert isinstance(y, np.ndarray)


@pytest.mark.parametrize("F", [BinaryClassifier, BinaryClassifierLPop])
def test_train(F):
    """Very basic call as the function can be v expensive"""
    data_dim = 1
    data_size = 10
    data = np.random.rand(data_size, data_dim)

    labels = np.random.randint(0, 2, size=(data_size))
    model = F(data_dim)
    y_start = model.predict(data)

    model = train(model, data, labels, num_epochs=1)
    y_end = model.predict(data)
    assert (y_start != y_end).any()
