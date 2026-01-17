import torch
import torch.nn as nn
from skimage.color import rgb2gray
from sklearn.base import BaseEstimator, ClassifierMixin




### Original network

class NewClassifyNet(nn.Module):
    def __init__(self):
        super(NewClassifyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1,64)),
            nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=5),
            nn.Sigmoid()    # wyłączyć do treningu i wizualizacji
        )

    def forward(self, x):
        return self.net(x)



# Classifier with no elipses

class NewClassifyNet4(nn.Module):
    def __init__(self):
        super(NewClassifyNet4, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1,64)),
            nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=4),
            nn.Sigmoid()    # wyłączyć do treningu i wizualizacji
        )

    def forward(self, x):
        return self.net(x)




### Original Wrapper

class NewWrapper(nn.Module):
    def __init__(self):
        super(NewWrapper, self).__init__()
        self.net = NewClassifyNet()





### Wrapper so it works with lime

class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, weight_path, device='cpu'):
        self.model_class = model_class
        self.weight_path = weight_path
        self.device = device
        self.model = None

    # ... (Other methods like _load_model, fit, predict remain unchanged) ...

    def _load_model(self):
        if self.model is None:
            self.model = self.model_class().to(self.device)
            state = torch.load(self.weight_path, map_location=self.device)

            # Handle various checkpoint formats
            if isinstance(state, dict):
                if 'state_dict' in state:
                    state = state['state_dict']
                elif 'model_state_dict' in state:
                    state = state['model_state_dict']

            # Fix double "net.net." prefix if needed
            new_state = {}
            for k, v in state.items():
                if k.startswith("net.net."):
                    new_state[k.replace("net.net.", "net.", 1)] = v
                else:
                    new_state[k] = v
            state = new_state

            self.model.load_state_dict(state, strict=False)
            self.model.eval()


    def fit(self, X=None, y=None):
        # just load model — no training
        self._load_model()
        return self

    def _prepare_input(self, X):
        gray_numpy = rgb2gray(X) # (64, 64, 3) -> (64, 64)
        gray_tensor = torch.from_numpy(gray_numpy).float() # (64, 64)
        return gray_tensor # Shape: (64, 64)

    def predict_proba(self, X):
        self._load_model()
        X_tensor = self._prepare_input(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs

    def predict(self, X):
        self._load_model()
        X_tensor = self._prepare_input(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs