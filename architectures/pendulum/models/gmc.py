from collections import Counter
from ..subnetworks.gmc_networks import *
from pytorch_lightning import LightningModule


class GMC(LightningModule):
    def __init__(self, name, exclude_modality, common_dim, latent_dim, infonce_temperature, loss_type="infonce"):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.exclude_modality = exclude_modality
        self.infonce_temperature = infonce_temperature

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = {
            'image_t': self.image_processor,
            'audio_t': self.label_processor,
            'joint': self.joint_processor,
        }

        self.encoder = None

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def encode(self, x, sample=False):
        # If we have complete observations
        if self.exclude_modality == 'none' or self.exclude_modality is None:
            return self.encoder(self.processors['joint'](x))
        else:
            latent_representations = []
            for key in x.keys():
                if key != self.exclude_modality:
                    latent_representations.append(self.encoder(self.processors[key](x[key])))

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
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.infonce_temperature
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
                / self.infonce_temperature
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

    def infonce_with_joints_as_negatives(self, batch_representations, batch_size):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / self.infonce_temperature
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
                / self.infonce_temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, labels):
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)
        return loss, Counter(tqdm_dict)

    def validation_step(self, data, labels):
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)
        return loss, Counter(tqdm_dict)


class PendulumGMC(GMC):
    def __init__(self, name, exclude_modality, common_dim, latent_dim, infonce_temperature, loss_type="infonce"):
        super(PendulumGMC, self).__init__(name, exclude_modality, common_dim, latent_dim, infonce_temperature, loss_type="infonce")
        self.loss_type = loss_type
        self.image_processor = PendulumImageProcessor(common_dim=common_dim)
        self.sound_processor = PendulumSoundProcessor(common_dim=common_dim)
        self.joint_processor = PendulumJointProcessor(common_dim=common_dim)
        if exclude_modality == 'image_t':
            self.processors = {'audio_t': self.sound_processor}
        elif exclude_modality == 'audio_t':
            self.processors = {'image_t': self.image_processor}
        else:
            self.processors = {
                'image_t': self.image_processor,
                'audio_t': self.sound_processor,
                'joint': self.joint_processor
            }

        self.encoder = PendulumCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)

    def set_common_dim(self, common_dim):
        self.encoder.set_common_dim(common_dim)
        for proc in self.processors.values():
            proc.set_common_dim(common_dim)