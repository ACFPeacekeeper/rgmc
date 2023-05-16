from pytorch_lightning import LightningModule
from architectures.gmc_networks import *
from collections import Counter

# Code adapted from https://github.com/miguelsvasco/gmc
class GMC(LightningModule):
    def __init__(self, name, common_dim, exclude_modality, latent_dim, loss_type="infonce"):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.exclude_modality = exclude_modality

        self.image_processor = None
        self.trajectory_processor = None
        self.joint_processor = None
        if self.exclude_modality == 'image':
            self.processors = {'trajectory': self.trajectory_processor}
        elif self.exclude_modality == 'trajectory':
            self.processors = {'image': self.image_processor}
        else:
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor,
            }

        self.encoder = None

    def encode(self, x, sample=False):
        if self.exclude_modality == 'none':
            return self.encoder(self.processors['joint'](x))
        else:
            latent_representations = []
            for key in x.keys():
                latent_representations.append(self.encoder(self.processors[key](x[key])))

            # Take the average of the latent representations
            if len(latent_representations) > 1:
                latent = torch.stack(latent_representations, dim=0).mean(0)
            else:
                latent_representations = latent_representations[0]
            return latent

    def forward(self, x):

        # Forward pass through the modality specific encoders
        batch_representations = []
        for key in x.keys():
            mod_representations = self.encoder(
                self.processors[key](x[key])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        if self.exclude_modality == 'none':
            joint_representation = self.encoder(self.processors['joint'](x))
            batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        mod_idx = len(batch_representations) - 1 if self.exclude_modality == 'none' else len(batch_representations)
        for mod in range(mod_idx):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
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
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def infonce_with_joints_as_negatives(
        self, batch_representations, temperature, batch_size
    ):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / temperature
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
                / temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params, batch_size):

        temperature = train_params["temperature"]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return loss, Counter(tqdm_dict)

    def validation_step(self, data, train_params, batch_size):

        temperature = train_params["temperature"]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return Counter(tqdm_dict)



class MhdGMC(GMC):
    def __init__(self, name, exclude_modality, latent_dim, loss_type="infonce"):
        if exclude_modality == 'image':
            self.common_dim = 200
        elif exclude_modality == 'trajectory':
            self.common_dim = 28 * 28
        else:
            self.common_dim = 28 * 28 + 200

        super(MhdGMC, self).__init__(name, self.common_dim, exclude_modality, latent_dim, loss_type)

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
        self.encoder = MHDCommonEncoder(common_dim=self.common_dim, latent_dim=latent_dim)

    def set_modalities(self, exclude_modality):
        if exclude_modality == 'image':
            self.common_dim = 200
            self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
            self.processors = {'trajectory': self.trajectory_processor}
        elif exclude_modality == 'trajectory':
            self.common_dim = 28 * 28
            self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
            self.processors = {'image': self.image_processor}
        else:
            self.common_dim = 28 * 28 + 200
            self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
            self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
            self.joint_processor = MHDJointProcessor(common_dim=self.common_dim)
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor
            }

        self.encoder = MHDCommonEncoder(common_dim=self.common_dim, latent_dim=self.latent_dim)
        self.exclude_modality = exclude_modality