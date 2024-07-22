from typing import List

import torch
import torch.nn as nn


class Hierarchical_Loss(nn.Module):
    def __init__(self, w1: float, w2: float, w3: float, weight: torch.tensor):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.weights = weight.float().cuda()
        self.sp_loss = nn.NLLLoss(weight=self.weights)
        self.ge_loss = nn.NLLLoss()
        self.fa_loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-6

    def list_unsqueezer(self, tensors_list: List) -> List:
        for i, tensor in enumerate(tensors_list):
            if len(tensor.shape) == 3:
                tensors_list[i] = tensor.unsqueeze(1)
        return tensors_list

    def species_level_aggregator(self, species_tensor: torch.tensor) -> List:
        species_tensor_list = [
            species_tensor[:, 0, :, :],
            species_tensor[:, 1, :, :],
            species_tensor[:, 2:6, :, :],
            species_tensor[:, 6:8, :, :],
            species_tensor[:, 8, :, :],
            species_tensor[:, 9, :, :],
            species_tensor[:, 10, :, :],
            species_tensor[:, 11, :, :],
            species_tensor[:, 12, :, :],
            species_tensor[:, 13, :, :],
            species_tensor[:, 14, :, :],
            species_tensor[:, 15, :, :],
        ]

        return species_tensor_list

    def genus_level_aggregator(self, genus_tensor: torch.tensor) -> List:
        background = genus_tensor[:, 0, :, :]
        conifer_fa = (
            genus_tensor[:, 1, :, :]
            + genus_tensor[:, 5, :, :]
            + genus_tensor[:, 7, :, :]
            + genus_tensor[:, 8, :, :]
            + genus_tensor[:, 10, :, :]
            + genus_tensor[:, 11, :, :]
        )
        nonconifera_fa = (
            genus_tensor[:, 2, :, :]
            + genus_tensor[:, 3, :, :]
            + genus_tensor[:, 4, :, :]
            + genus_tensor[:, 9, :, :]
        )
#        palm_fa = genus_tensor[:, 5, :, :]
        dead_fa = genus_tensor[:, 6, :, :]

        family_tensor_list = [background, conifer_fa, nonconifera_fa, dead_fa] # Removing palm_fa
        return family_tensor_list

    def aggregate_probabilites_species(self, tensors_list: List) -> torch.tensor:
        first_el = tensors_list[0]
        if len(first_el.shape) == 4 and first_el.shape[1] > 1:
            first_el = torch.sum(first_el, dim=1).unsqueeze(1)

        for tensor in tensors_list[1:]:
            if len(tensor.shape) == 4 and tensor.shape[1] > 1:
                tensor = torch.sum(tensor, dim=1).unsqueeze(1)
            first_el = torch.cat((first_el, tensor), dim=1)
        return first_el

    def aggregate_probabilites_genus(self, genus_tensor_list: List) -> torch.tensor:
        fa_tensor = torch.cat(
            (
                genus_tensor_list[0],
                genus_tensor_list[1],
                genus_tensor_list[2],
                genus_tensor_list[3],
#                genus_tensor_list[4], #Not needed anymore
            ),
            dim=1,
        )
        return fa_tensor

    def forward(
        self, output: torch.tensor, target: torch.tensor
    ) -> torch.tensor:  # Change output type List[torch.tensor, torch.tensor, torch.tensor]
        # Get the log softmax probabilities
        species_tensor = self.softmax(output)

        # Species level list
        species_tensor_list = self.species_level_aggregator(species_tensor)
        # Unsqueeze tensors in the list
        species_tensor_list = self.list_unsqueezer(species_tensor_list)
        # Species level tensor
        genus_tensor = self.aggregate_probabilites_species(species_tensor_list)

        # Family level list
        genus_tensor_list = self.genus_level_aggregator(genus_tensor)
        # Unsqueeze tensors in the list
        genus_tensor_list = self.list_unsqueezer(genus_tensor_list)
        # Family level tensor
        family_tensor = self.aggregate_probabilites_genus(genus_tensor_list)

        # Doing torch.log on the softmax probabilities instead of using logsoftmax for
        # metric calculation using the genus and family tensors.
        #        print(target[:, 0, :, :].type(), species_tensor.type())
        sp_loss_tensor = self.w1 * self.sp_loss(
            torch.log(species_tensor + self.epsilon), target[:, 0, :, :]
        )
        ge_loss_tensor = self.w2 * self.ge_loss(
            torch.log(genus_tensor + self.epsilon), target[:, 1, :, :]
        )
        fa_loss_tensor = self.w3 * self.fa_loss(
            torch.log(family_tensor + self.epsilon), target[:, 2, :, :]
        )

        #        print(
        #            "Species Loss: {}, Genus Loss: {}, Family loss: {}".format(
        #                sp_loss_tensor.item(), ge_loss_tensor.item(), fa_loss_tensor.item()
        #            )
        #        )

        loss = sp_loss_tensor + 0.3 * ge_loss_tensor + 0.1 * fa_loss_tensor
        return (
            loss,
            sp_loss_tensor.item(),
            ge_loss_tensor.item(),
            fa_loss_tensor.item(),
            genus_tensor,
            family_tensor,
        )
