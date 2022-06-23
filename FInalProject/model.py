import torch.nn as nn

class feature_mixer(nn.Module) :
    def __init__(self, io_size) :
        super(feature_mixer, self).__init__()
        self.conv1 = nn.Conv2d(io_size, io_size, kernel_size=5, stride=1, padding=2, groups=io_size)
        self.conv2 = nn.Conv2d(io_size, io_size, kernel_size=7, stride=1, padding=3, groups=io_size)
        self.conv3 = nn.Conv2d(io_size, io_size, kernel_size=9, stride=1, padding=4, groups=io_size)
        self.relu = nn.ReLU()

    def forward(self, x) :
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.relu(x1 + x2 + x3)

class conv_autoencoder(nn.Module):
    def __init__(self, dr=0.2, concat_n=5):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(concat_n, 120, 4, stride=2, padding=1, groups=concat_n),      
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Dropout2d(dr),

            nn.Conv2d(120, 240, 4, stride=2, padding=1),  
            nn.BatchNorm2d(240),      
            nn.ReLU(),
            nn.Dropout2d(dr),
            
			nn.Conv2d(240, 480, 4, stride=2, padding=1),
            nn.BatchNorm2d(480),
            nn.ReLU(),
            nn.Dropout2d(dr),

            feature_mixer(480),
            nn.BatchNorm2d(480),
            nn.ReLU(), 
            nn.Dropout2d(dr),
        )

        self.decoder = nn.Sequential(
    	    nn.ConvTranspose2d(480, 128, 4, stride=2, padding=1), 
            nn.ReLU(),

			nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Tanh()
        )

        

    def forward(self, x):
        x2 = self.encoder(x)
        #x1 = self.binary_classifier_prenet(x2)
        #x1 = x1.flatten(start_dim=1)
        #x1 = self.binary_classifier(x1)
        x2 = self.decoder(x2)
        #x2 = torch.cat((x2, x), dim=1)
        #x2 = self.filter(x2)
        return x2# , x1


class binary_classifier(nn.Module) :
    def __init__(self, concat_n = 3):
        super(binary_classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(concat_n, 120, 4, stride=2, padding=1, groups=concat_n),         
            nn.ReLU(),

            nn.Conv2d(120, 256, 4, stride=2, padding=1),        
            nn.ReLU(),
            
			nn.Conv2d(256, 512, 4, stride=2, padding=1),         
            nn.ReLU(),

            feature_mixer(512),
            nn.ReLU(), 
        )
        
        self.binary_classifier_prenet = nn.Sequential(
            nn.Conv2d(512, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.binary_classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x) :
        x = self.encoder(x)
        x = self.binary_classifier_prenet(x)
        x = x.view(x.size(0), -1)
        x = self.binary_classifier(x)
        return x