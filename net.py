import torch
import torch.nn as nn
import torch.nn.functional as F

class dogs_stanford(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()
        self.seq = nn.Sequential(
        #3 levels, first for general recognition, second for shapes,etc, the third for small details
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.MaxPool2d(2, 2), #weight and height/2 for better performance, second and third input is 32
        nn.Dropout(0.4), #we ignore 0.4 neurons randomly to prevent overfitting

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Dropout(0.1),


        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ELU()
        )
        self.fc1 = nn.Linear(128, 256) #a modificar
        self.fc2 = nn.Linear(256, num_classes)
    
        #init kaiming for better performance
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.seq(x)
        x = F.adaptive_avg_pool2d(x, (1,1))  # it makes an approximation of the original output
        #it sacrifies some accuracy for performance
        x = torch.flatten(x, 1) #from R3 -> R1
        x = self.fc1(x)
        x = self.fc2(x)
        return x


