from torch import nn


class Quality(nn.Module):
    def __init__(self, action_count=3):
        super(Quality, self).__init__()

        self.stack_of_layers = nn.Sequential(
            nn.Linear(2, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, action_count)
        )

    def forward(self, x):
        logits = self.stack_of_layers(x)

        return logits
