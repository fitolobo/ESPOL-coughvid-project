import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# Añadir StandardScaler a la lista de clases seguras
torch.serialization.add_safe_globals([StandardScaler])


class CombinedDataset(Dataset):
    """
        Clase para almacenar y validar los tipos de datos generados
        en la etapa de preprocessing.
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, scalar_features, spectrograms, chromagrams, labels):
        self.scalar_features = torch.FloatTensor(scalar_features)
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.chromagrams = torch.FloatTensor(chromagrams)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.scalar_features[idx],
            self.spectrograms[idx],
            self.chromagrams[idx],
            self.labels[idx],
        )


class CoughNetWithCNN(torch.nn.Module):
    """Modelo Híbrido de Red Convolucional
       y DNN para la identificación de casos de Covid-19
    """
    def __init__(self, scalar_input_size):
        super(CoughNetWithCNN, self).__init__()

        # CNN for spectrogram (input: batch x 1 x 128 x 128)
        self.spec_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16 x 64 x 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 64 x 4 x 4
        )

        # CNN for chromagram (input: batch x 1 x 12 x 128)
        self.chroma_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16 x 6 x 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),  # -> 32 x 2 x 4
        )

        # Process scalar features
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_size, 64), nn.ReLU(), nn.Dropout(0.5)
        )

        # Final layers - combine all features
        spec_features = 64 * 4 * 4  # From spec_conv
        chroma_features = 32 * 2 * 4  # From chroma_conv
        combined_size = 64 + spec_features + chroma_features  # 64 from scalar_net

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, scalar_x, spectrogram, chromagram):
        # Process scalar features
        scalar_features = self.scalar_net(scalar_x)

        # Process spectrogram - add channel dimension if needed

        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram.unsqueeze(1)
        spec_features = self.spec_conv(spectrogram)
        spec_features = spec_features.view(spec_features.size(0), -1)

        # Process chromagram - add channel dimension if needed
        if len(chromagram.shape) == 3:
            chromagram = chromagram.unsqueeze(1)
        chroma_features = self.chroma_conv(chromagram)
        chroma_features = chroma_features.view(chroma_features.size(0), -1)

        # Combine all features
        combined = torch.cat([scalar_features, spec_features, chroma_features], dim=1)

        # Final classification
        return self.classifier(combined)


def prepare_data_loaders(
    scalar_features, spectrograms, chromagrams, labels, batch_size=32
):
    # First create train/val indices
    dataset_size = len(labels)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Scale scalar features using only training data
    scaler = StandardScaler()
    scalar_features_train = scaler.fit_transform(scalar_features[train_indices])
    scalar_features_val = scaler.transform(scalar_features[val_indices])

    # Create separate datasets for train and validation
    train_dataset = CombinedDataset(
        scalar_features_train,
        spectrograms[train_indices],
        chromagrams[train_indices],
        labels[train_indices],
    )

    val_dataset = CombinedDataset(
        scalar_features_val,
        spectrograms[val_indices],
        chromagrams[val_indices],
        labels[val_indices],
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler