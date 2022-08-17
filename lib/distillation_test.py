import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import DML
from KD_Lib.models import ResNet18, ResNet50          # To use models packaged in KD_Lib

# Define your datasets, dataloaders, models and optimizers

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

student_params = [4, 4, 4, 4, 4]
student_model_1 = ResNet50(student_params, 1, 10)
student_model_2 = ResNet18(student_params, 1, 10)

student_cohort = [student_model_1, student_model_2]

student_optimizer_1 = optim.SGD(student_model_1.parameters(), 0.01)
student_optimizer_2 = optim.SGD(student_model_2.parameters(), 0.01)

student_optimizers = [student_optimizer_1, student_optimizer_2]

# Now, this is where KD_Lib comes into the picture

distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, log=True, logdir="./logs")

distiller.train_students(epochs=5)
distiller.evaluate()
distiller.get_parameters()