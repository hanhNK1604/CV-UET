import torch 
from torch import nn 

class VGG16(nn.Module): 
    def __init__(
        self, 
        config = [3, 64, 128, 256, 512, 512] 
    ): 
        super(VGG16, self).__init__() 
        self.config = config 

        self.main_archi = nn.ModuleList() 
        for i in range(len(self.config)): 
            if i == 0: 
                continue 
            else: 
                conv1 = nn.Conv2d(in_channels=self.config[i - 1], out_channels=self.config[i], kernel_size=3, padding=1) 
                relu1 = nn.ReLU(inplace=True) 
                conv2 = nn.Conv2d(in_channels=self.config[i], out_channels=self.config[i], kernel_size=3, padding=1) 
                relu2 = nn.ReLU(inplace=True)
                max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
                
                self.main_archi.append(conv1) 
                self.main_archi.append(relu1) 
                self.main_archi.append(conv2) 
                self.main_archi.append(relu2) 
                self.main_archi.append(max_pool)
                
        self.linear = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096), 
            nn.Linear(in_features=4096, out_features=4096), 
            nn.Linear(in_features=4096, out_features=1000) 
        )

    def forward(self, x): 
        for layer in self.main_archi: 
            x = layer(x) 
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x) 

        return x 

# a = torch.rand(4, 3, 224, 224) 
# net = VGG16() 
# print(net(a).shape) 
