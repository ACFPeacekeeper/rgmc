import random

from ..subnetworks.ooo_network import OddOneOutNetwork
from pytorch_lightning import LightningModule
from ..subnetworks.rgmc_networks import *
from collections import Counter


class RGMC(LightningModule):
    def __init__(self, name, common_dim, exclude_modality, latent_dimension, scales, noise_factor=0.3, loss_type="infonce"):
        super(RGMC, self).__init__()
        self.name = name
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.loss_type = loss_type
        self.exclude_modality = exclude_modality
        self.scales = scales
        self.noise_factor = noise_factor

        self.image_processor = None
        self.trajectory_processor = None
        self.joint_processor = None
        if self.exclude_modality == 'image':
            self.num_modalities = 1
            self.modalities = ["trajectory"]
            self.processors = {'trajectory': self.trajectory_processor}
            self.num_modalities = 1
        elif self.exclude_modality == 'trajectory':
            self.num_modalities = 1
            self.modalities = ["image"]
            self.processors = {'image': self.image_processor}
        else:
            self.num_modalities = 2
            self.modalities = ["image", "trajectory"]
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor,
            }

        self.encoder = None
        self.o3n = None

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def add_perturbation(self, x):
        target_modality = self.modalities[random.randint(0, self.num_modalities - 1)]
        for key, modality in x.items():
            if key == target_modality:
                x[key] = torch.clamp(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), torch.min(modality), torch.max(modality))
            else:
                x[key] = modality
        return x

    def encode(self, x, sample=False):
        latent_representations = []
        for key in x.keys():
            if key != self.exclude_modality:
                latent_representations.append(self.encoder(self.processors[key](x[key])))

        mod_weights = self.o3n(latent_representations)
        print(mod_weights[0])

        for id, latent_repr in enumerate(latent_representations):
            latent_representations[id] = torch.mul(latent_repr, mod_weights[:, id])

        # Take the average of the latent representations
        if len(latent_representations) > 1:
            latent = torch.stack(latent_representations, dim=0).mean(0)
        else:
            latent = latent_representations[0]
        return latent

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for key in x.keys():
            if key != self.exclude_modality:
                mod_representations = self.encoder(
                    self.processors[key](x[key])
                )
                batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        if self.exclude_modality == 'none' or self.exclude_modality is None:
            joint_representation = self.encoder(self.processors['joint'](x))
            batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, batch_size):
        joint_mod_loss_sum = 0
        mod_idx = len(batch_representations) - 1 if (self.exclude_modality == 'none' or self.exclude_modality is None) else len(batch_representations)
        for mod in range(mod_idx):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.scales['infonce_temp']
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / self.scales['infonce_temp']
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"infonce_loss": loss}
        return loss, tqdm_dict

    def infonce_with_joints_as_negatives(self, batch_representations, batch_size):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / self.scales['infonce_temp']
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
            torch.ones_like(sim_matrix_joints)
            - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(
            batch_size, -1
        )

        # compute loss - for each pair joints-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / self.scales['infonce_temp']
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"infonce_loss": loss}
        return loss, tqdm_dict
    
    def o3n_loss(self, perturbed_preds, clean_pred):
        clean_pred = clean_pred.view(clean_pred.size(0), 1)
        preds = torch.cat((perturbed_preds, clean_pred), dim=-1)
        loss = - torch.mean(torch.log(preds)) * self.scales["o3n_loss"]
        return loss, {"odd_one_out_loss:": loss}

    def training_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Forward pass through odd-one-out network
        clean_preds = self.o3n(batch_representations[:-1])
        perturbed_batch = self.add_perturbation(data)
        perturbed_reps = self.forward(perturbed_batch)
        perturbed_preds = self.o3n(perturbed_reps[:-1])
        o3n_loss, o3n_dict = self.o3n_loss(perturbed_preds[:, :-1], clean_preds[:, -1])

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)

        total_loss = loss + o3n_loss
        return total_loss, Counter({"total_loss": total_loss, **tqdm_dict, **o3n_dict})

    def validation_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Forward pass through odd-one-out network
        clean_preds = self.o3n(batch_representations[:-1])
        perturbed_batch = self.add_perturbation(data)
        perturbed_reps = self.forward(perturbed_batch)
        perturbed_preds = self.o3n(perturbed_reps[:-1])
        o3n_loss, o3n_dict = self.o3n_loss(perturbed_preds[:, :-1], clean_preds[:, -1])

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)
        
        total_loss = loss + o3n_loss
        return total_loss, Counter({"total_loss": total_loss, **tqdm_dict, **o3n_dict})



class MhdRGMC(RGMC):
    def __init__(self, name, exclude_modality, common_dim, latent_dimension, scales, noise_factor, device, loss_type="infonce"):
        self.common_dim = common_dim

        super(MhdRGMC, self).__init__(name, self.common_dim, exclude_modality, latent_dimension, scales, noise_factor, loss_type)

        self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
        self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
        self.joint_processor = MHDJointProcessor(common_dim=self.common_dim)

        if exclude_modality == 'image':
            self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
            self.processors = {'trajectory': self.trajectory_processor}
        elif exclude_modality == 'trajectory':
            self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
            self.processors = {'image': self.image_processor}
        else:
            self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
            self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
            self.joint_processor = MHDJointProcessor(common_dim=self.common_dim)
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor
            }

        self.loss_type = loss_type
        self.encoder = MHDCommonEncoder(common_dim=self.common_dim, latent_dimension=latent_dimension)
        self.o3n = OddOneOutNetwork(latent_dim=self.latent_dimension, num_modalities=2, device=device)

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality
        