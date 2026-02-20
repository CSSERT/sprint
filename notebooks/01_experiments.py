# %% Libraries
import backend.models.bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from backend.services import DataService, ForecastingModelService, TrainerService

# %% Load data
feature_cols = ["volume", "close"]
target_col = "close"
data_service = DataService(
    {
        "data_dir": "../data/raw",
        "interval": "daily",
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
)
train_loader, test_loader = data_service.get("VCB")

# %% Model
lstm = ForecastingModelService(
    "sprint.rnn.lstm",
    {
        "input_size": len(feature_cols),
        "output_size": 1,
    },
)

# %% Training
trainer = TrainerService(
    forecaster=lstm,
    optimizer=optim.AdamW(lstm.model.parameters(), lr=1e-3),
    criterion=nn.MSELoss(),
)
trainer.train(
    train_loader,
    epochs=10,
)

# %% Evaluation
y_trues, y_hats = [], []
for x, y in test_loader:
    y_trues.extend(y.cpu().numpy().tolist())
    y_hats.extend(lstm.predict(x.to("cuda")).cpu().numpy().tolist())
plt.plot(y_trues)
plt.plot(y_hats)
plt.show()
