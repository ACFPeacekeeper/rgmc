from pytorch_lightning import LightningModule
from ..subnetworks.dgmc_networks import *
from collections import Counter


class DGMC(LightningModule):
    def __init__(self, name, common_dim, exclude_modality, latent_dimension, scales, loss_type="infonce", noise_factor=1.0):
        super(DGMC, self).__init__()
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
        self.image_reconstructor = None
        self.trajectory_reconstructor = None
        self.joint_reconstructor = None
        if self.exclude_modality == 'image':
            self.processors = {'trajectory': self.trajectory_processor}
            self.reconstructors = {'trajectory': self.trajectory_reconstructor}
        elif self.exclude_modality == 'trajectory':
            self.processors = {'image': self.image_processor}
            self.reconstructors = {'image': self.image_reconstructor}
        else:
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor,
            }
            self.reconstructors = {
                'image': self.image_reconstructor,
                'trajectory': self.trajectory_reconstructor,
                'joint': self.joint_reconstructor,
            }

        self.encoder = None
        self.decoder = None

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def add_noise(self, x):
        x_noisy = dict.fromkeys(x.keys())
        for key, modality in x.items():
            x_noisy[key] = torch.clamp(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), torch.min(modality), torch.max(modality))
        return x_noisy

    def encode(self, x, sample=False):
        if sample is False:
            x = self.add_noise(x)

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
        
    def decode(self, z):
        if self.exclude_modality == 'none' or self.exclude_modality is None:
            #reconstructions['joint'] = self.reconstructors['joint'](self.decoder(z))
            return self.reconstructors['joint'](self.decoder(z[-1]))

        reconstructions = dict.fromkeys(z.keys())
        for key, mod_id in enumerate(reconstructions.keys()):
            if key != self.exclude_modality:
                reconstructions[key] = self.reconstructors[key](self.decoder(z[mod_id]))

        return reconstructions

    def forward(self, x, sample=False):
        if sample is False:
            x = self.add_noise(x)

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

    def recon_loss(self, x, z):
        x_hat = self.decode(z)

        mse_loss = nn.MSELoss(reduction="none").to(self.device)
        recon_losses = dict.fromkeys(x.keys())

        for key in recon_losses.keys():
            cost = mse_loss(x[key], x_hat[key])
            recon_losses[key] = self.scales[key] * (cost / torch.as_tensor(cost.size()).prod().sqrt()).sum() 

        loss = sum(recon_losses.values()) / len(recon_losses)

        return loss, {'image_recon_loss': recon_losses['image'], 'traj_recon_loss': recon_losses['trajectory']}

    def training_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)

        recon_loss, recon_dict = self.recon_loss(data, batch_representations)
        total_loss = loss + recon_loss
        loss_dict = {'total_loss': total_loss, **tqdm_dict, **recon_dict}
        return total_loss, Counter(loss_dict)

    def validation_step(self, data, labels):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, batch_size)
        
        recon_loss, recon_dict = self.recon_loss(data, batch_representations)
        total_loss = loss + recon_loss
        loss_dict = {'total_loss': total_loss, **tqdm_dict, **recon_dict}
        return loss, Counter(loss_dict)



class MhdDGMC(DGMC):
    def __init__(self, name, exclude_modality, latent_dimension, infonce_temperature, loss_type="infonce"):
        if exclude_modality == 'image':
            self.common_dim = 200
        elif exclude_modality == 'trajectory':
            self.common_dim = 28 * 28
        else:
            self.common_dim = 28 * 28 + 200

        super(MhdDGMC, self).__init__(name, self.common_dim, exclude_modality, latent_dimension, infonce_temperature, loss_type)

        self.image_processor = MHDImageProcessor(common_dim=self.common_dim)
        self.trajectory_processor = MHDTrajectoryProcessor(common_dim=self.common_dim)
        self.joint_processor = MHDJointProcessor(common_dim=self.common_dim)
        self.image_reconstructor = MHDImageDecoder(common_dim=self.common_dim)
        self.trajectory_reconstructor = MHDTrajectoryDecoder(common_dim=self.common_dim)
        self.joint_reconstructor = MHDJointDecoder(common_dim=self.common_dim)

        if exclude_modality == 'image':
            self.processors = {'trajectory': self.trajectory_processor}
            self.reconstructors = {'trajectory': self.trajectory_reconstructor}
        elif exclude_modality == 'trajectory':
            self.processors = {'image': self.image_processor}
            self.reconstructors = {'image': self.image_reconstructor}
        else:
            self.processors = {
                'image': self.image_processor,
                'trajectory': self.trajectory_processor,
                'joint': self.joint_processor,
            }
            self.reconstructors = {
                'image': self.image_reconstructor,
                'trajectory': self.trajectory_reconstructor,
                'joint': self.joint_reconstructor,
            }

        self.loss_type = loss_type
        self.encoder = MHDCommonEncoder(common_dim=self.common_dim, latent_dimension=latent_dimension)
        self.decoder = MHDCommonDecoder(common_dim=self.common_dim, latent_dimension=latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        self.decoder.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality
        