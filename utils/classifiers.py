import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out = F.relu(self.conv2(out))
        out = out.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), dim=1)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return F.log_softmax(self.fc3(out), dim=1)


MODELS = {"cnn": ConvNet, "mlp": SimpleMLP}


def load_classifier(kind, path, device="cpu"):
    model = MODELS[kind]().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, x_0_255, device="cpu", batch_size=512):
    """``x_0_255``: numpy (N,28,28) with values in 0..255. Returns predicted labels (N,)."""
    xt = torch.as_tensor(x_0_255, dtype=torch.float32).unsqueeze(1)
    preds = []
    for i in range(0, len(xt), batch_size):
        preds.append(model(xt[i:i + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()
