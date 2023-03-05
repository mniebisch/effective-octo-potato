"""
File to train the first, very simple example.

author = Christian Reimers
"""
import pathlib

import torch
import tqdm

from effective_octo_potato import SimpleNet
from effective_octo_potato.data import LandmarkDataset

if __name__ == '__main__':

    nr_epochs = 8

    train_dataset = LandmarkDataset(
            pathlib.Path('../data'),
            pathlib.Path('train.csv'),
            pathlib.Path('sign_to_prediction_index_map.json'),
            ignore_z = True,
        )
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size = 1, shuffle = True)

    classifier = SimpleNet(nr_inputs = 543 * 2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max = 100,
            )

    for _ in range(nr_epochs):
        epoch_loss = 5.52 # ln(250)
        correct = 0
        combined = 0
        examples = tqdm.tqdm(train_dataloader)
        for batch, label in examples:
            optimizer.zero_grad()
            prediction = classifier(batch)
            loss = criterion(prediction, label)
            epoch_loss = 0.9 * epoch_loss + 0.1 * loss.item()

            loss.backward()
            combined = combined + 1
            if torch.argmax(prediction) == label:
                correct = correct + 1

            examples.set_postfix({'epoch loss': epoch_loss, 'accuracy': correct / combined})

            optimizer.step()
        
        scheduler.step()

    
