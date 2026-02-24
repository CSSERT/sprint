# %% Libraries
import backend.models.bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import torch.optim as optim
from backend.losses import QuantileLoss
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
data_service = DataService(
    {
        "interval": "daily",
        "test_size": 0.2,
    }
)
train_loader, test_loader = data_service.get("VCB")

# %% Model
lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "n_features": 5,
        "n_horizons": 3,
        "n_quantiles": 3,
    },
)

# %% Training
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=QuantileLoss([0.1, 0.5, 0.9]),
)
trainer.train(
    train_loader,
    epochs=50,
)

# %% Evaluation
y_hats, y_trues = lstm.predict(test_loader, return_targets=True)
y_hats = data_service.inverse_y(y_hats)
y_trues = data_service.inverse_y(y_trues)
plt.plot(y_trues[:, 0])
plt.plot(y_hats[:, 0, 1])
plt.show()
