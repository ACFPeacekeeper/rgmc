import random, sys
import torch.nn.functional as F

from collections import Counter
from ..subnetworks.rgmc_networks import *
from pytorch_lightning import LightningModule
from ..subnetworks.ooo_network import OddOneOutNetwork

class RGMC(LightningModule):
    def __init__(self, name, common_dim, exclude_modality, latent_dimension, scales, noise_factor=0.3, loss_type="infonce"):
        super(RGMC, self).__init__()
        self.name = name
        self.scales = scales
        self.loss_type = loss_type
        self.common_dim = common_dim
        self.noise_factor = noise_factor
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.joint_reconstructor = None
        self.image_processor = None
        self.trajectory_processor = None
        self.joint_processor = None
        self.processors = {
            'image': self.image_processor,
            'trajectory': self.trajectory_processor,
            'joint': self.joint_processor,
        }
        if self.exclude_modality == 'image':
            self.num_modalities = 1
            self.modalities = ["trajectory"]
        elif self.exclude_modality == 'trajectory':
            self.num_modalities = 1
            self.modalities = ["image"]
        else:
            self.num_modalities = 2
            self.modalities = ["image", "trajectory"]
        self.encoder = None
        self.o3n = None
        self.perturbation = None

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality
        if self.exclude_modality == '"image"':
            self.num_modalities = 1
            self.modalities = ["trajectory"]
        elif self.exclude_modality == 'trajectory':
            self.num_modalities = 1
            self.modalities = ["image"]
        else:
            self.num_modalities = 2
            self.modalities = ["image", "trajectory"]

    def add_perturbation(self, x, y):
        # Last id corresponds to targetting none of the modalities
        target_id = random.randint(0, self.num_modalities)
        if target_id < 2:
            target_modality = self.modalities[target_id]
            self.perturbation._set_target_modality(target_modality)
            x = self.perturbation(x, y)
        return x, target_id

    def encode(self, x, sample=False):
        batch_size = list(x.values())[0].size(dim=0)
        if sample is False and self.noise_factor != 0:
            x = self.add_perturbation(x)

        latent_representations = []
        for key in x.keys():
            if key != self.exclude_modality:
                latent_representations.append(self.encoder(self.processors[key](x[key])))

        if self.exclude_modality == 'none' or self.exclude_modality is None:
            mod_weights = self.o3n(latent_representations, batch_size)
            latent_representations.append(self.encoder(self.processors['joint'](x)))
            for id, rep in enumerate(latent_representations):
                latent_representations[id] = torch.mul(rep, mod_weights[:, id])

        # Take the average of the latent representations
        if len(latent_representations) > 1:
            return torch.stack(latent_representations, dim=0).mean(dim=0)
        else:
            return latent_representations[0]

    def forward(self, x, y):
        clean_representations = []
        for key in x.keys():
            if key != self.exclude_modality:
                mod_representations = self.encoder(
                    self.processors[key](x[key])
                )
                clean_representations.append(mod_representations)
        
        # Forward pass through the joint encoder
        if self.exclude_modality == 'none' or self.exclude_modality is None:
            joint_representation = self.encoder(self.processors['joint'](x))
            clean_representations.append(joint_representation)

        x, target_id = self.add_perturbation(x, y)

        # Forward pass through the modality specific encoders
        batch_representations = []
        for key in x.keys():
            if key != self.exclude_modality:
                mod_representations = self.encoder(
                    self.processors[key](x[key])
                )
                batch_representations.append(mod_representations)

        return clean_representations, batch_representations, target_id

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
    
    def o3n_loss(self, perturbed_mod_weights, target_id, batch_size):
        ce_loss = nn.BCEWithLogitsLoss().to(self.device)
        target_labels = torch.full((batch_size,), target_id).to(self.device)
        target_labels_1hot = F.one_hot(target_labels, self.num_modalities + 1).float()
        loss = ce_loss(perturbed_mod_weights, target_labels_1hot) * self.scales['o3n_loss_scale']
        return loss, {"o3n_loss": loss}

    def training_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        clean_representations, batch_representations, target_id = self.forward(data, labels)

        # Forward pass through odd-one-out network
        perturbed_mod_weights = self.o3n(batch_representations)
        o3n_loss, o3n_dict = self.o3n_loss(perturbed_mod_weights, target_id, batch_size)

        for id, rep in enumerate(batch_representations):
            batch_representations[id] = torch.mul(rep, perturbed_mod_weights[:, id])
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(clean_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(clean_representations, batch_size)

        total_loss = loss + o3n_loss
        return total_loss, Counter({"total_loss": total_loss, **tqdm_dict, **o3n_dict})
    
    def validation_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        batch_representations = self.forward(data, sample=True)
         # Forward pass through odd-one-out network
        clean_mod_weights = self.o3n(batch_representations[:-1])
        perturbed_batch = self.add_perturbation(data)
        perturbed_reps = self.forward(perturbed_batch)
        perturbed_mod_weights = self.o3n(perturbed_reps[:-1])
        o3n_loss, o3n_dict = self.o3n_loss(perturbed_mod_weights[:, :-1], clean_mod_weights[:, -1])

        for id, rep in enumerate(perturbed_reps):
            perturbed_reps[:, id] = torch.mul(rep, perturbed_mod_weights[:, id])

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(perturbed_reps, batch_size)
        else:
            loss, tqdm_dict = self.infonce(perturbed_reps, batch_size)
        
        total_loss = loss + o3n_loss
        return total_loss, Counter({"total_loss": total_loss, **tqdm_dict, **o3n_dict})


class MhdRGMC(RGMC):
    def __init__(self, name, exclude_modality, common_dim, latent_dimension, scales, noise_factor, device, loss_type="infonce"):
        super(MhdRGMC, self).__init__(name, common_dim, exclude_modality, latent_dimension, scales, noise_factor, loss_type)
        self.traj_dim = 512
        self.image_dims = [128, 7, 7]
        image_dim = reduce(lambda x, y: x * y, self.image_dims)
        self.image_processor = MHDImageProcessor(common_dim=self.common_dim, dim=image_dim)
        self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim, dim=self.traj_dim)
        self.joint_processor = MHDJointProcessor(common_dim=self.common_dim, image_dim=image_dim, traj_dim=self.traj_dim)
        if self.exclude_modality == 'image':
            self.num_modalities = 1
            self.modalities = ["trajectory"]
        elif self.exclude_modality == 'trajectory':
            self.num_modalities = 1
            self.modalities = ["image"]
        else:
            self.num_modalities = 2
            self.modalities = ["image", "trajectory"]

        self.processors = {
            'image': self.image_processor,
            'trajectory': self.trajectory_processor,
            'joint': self.joint_processor,
        }
        self.loss_type = loss_type
        self.encoder = MHDCommonEncoder(common_dim=common_dim, latent_dimension=latent_dimension)
        self.o3n = OddOneOutNetwork(latent_dim=self.latent_dimension, num_modalities=self.num_modalities, modalities=self.modalities, device=device)
        self.perturbation = None

    def set_perturbation(self, perturbation):
        self.perturbation = perturbation

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        self.o3n.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def set_common_dim(self, common_dim):
        self.encoder.set_common_dim(common_dim)
        for proc in self.processors.values():
            proc.set_common_dim(common_dim)
        self.common_dim = common_dim

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality