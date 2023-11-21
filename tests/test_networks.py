import numpy as np
import pytest
import torch

from lsbi.network import BinaryClassifier, BinaryClassifierBase, BinaryClassifierLPop


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
        return torch.tensor(np.random.randint(0, 2, size=(10, 1)), dtype=torch.float32)

    def fit_model(self, model, input_dim):
        data_size = 10
        data = np.random.rand(data_size, input_dim)
        labels = np.random.randint(0, 2, size=(data_size))
        y_start = model.predict(data)
        model.fit(data, labels, num_epochs=1)
        y_end = model.predict(data)
        return y_start, y_end

    def test_init(self, model):
        assert isinstance(model, BinaryClassifierBase)

    def test_forward(self, model, x):
        y = model.forward(x)
        assert y.shape == (10, 1)

    def test_loss(self, model, x, y):
        with pytest.raises(NotImplementedError):
            model.loss(x, y)

    def test_predict(self, model, x):
        with pytest.raises(NotImplementedError):
            model.predict(x)

    def test_fit(self, model, x):
        with pytest.raises(NotImplementedError):
            y_start, y_end = self.fit_model(model, x.shape[1])


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

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_fit(self, model, x):
        y_start, y_end = self.fit_model(model, x.shape[1])
        assert (y_start != y_end).any()


@pytest.mark.parametrize("alpha", [2, 5])
class TestClassifierLPop(TestClassifierBase):
    CLS = BinaryClassifierLPop

    @pytest.fixture
    def model(self, input_dim, internal_dim, initial_dim, alpha):
        return self.CLS(input_dim, internal_dim, initial_dim, alpha=alpha)

    def test_loss(self, model, x, y):
        loss = model.loss(x, y)
        assert loss.detach().numpy().shape == ()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_predict(self, model, x):
        y = model.predict(x)
        assert y.shape == (10, 1)
        assert isinstance(y, np.ndarray)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_fit(self, model, x):
        y_start, y_end = self.fit_model(model, x.shape[1])
        assert (y_start != y_end).any()
