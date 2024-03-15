from ..subnetworks.gmc_networks import *
from collections import Counter

# Code adapted from https://github.com/miguelsvasco/gmc
class SuperGMC(LightningModule):
    def __init__(self, name, common_dim, exclude_modality, latent_dim, infonce_temperature, loss_type="infonce"):
        super(SuperGMC, self).__init__()

        self.name = name
        self.loss_type = loss_type
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.exclude_modality = exclude_modality
        self.infonce_temperature = infonce_temperature

        self.vision_processor = None
        self.language_processor = None
        self.joint_processor = None
        self.processors = {
            'vision': self.vision_processor,
            'text': self.language_processor,
            'joint': self.joint_processor,
        }

        self.encoder = None
        self.criterion = nn.L1Loss()

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def encode(self, x, sample=False):
        # If we have complete observations
        if self.exclude_modality == 'none' or self.exclude_modality is None:
            joint_representation = self.encoder(self.processors['joint'](x))

            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
            output += joint_representation

            return self.classifier(output)

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

            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(latent)), p=0.0, training=self.training))
            output += latent

            return self.classifier(output)

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

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
        output += joint_representation

        return self.classifier(output), batch_representations

    def super_gmc_loss(self, prediction, target, batch_representations, batch_size):
        joint_mod_loss_sum = 0
        mod_idx = len(batch_representations) - 1 if (self.exclude_modality == 'none' or self.exclude_modality is None) else len(batch_representations)
        for mod in range(mod_idx):
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

        supervised_loss = self.criterion(prediction, target)

        loss = torch.mean(joint_mod_loss_sum + supervised_loss)
        # loss = torch.mean(supervised_loss)
        tqdm_dict = {"infonce_loss": loss}
        return loss, tqdm_dict


    def training_step(self, data, target_data):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, batch_size)

        return loss, tqdm_dict

    def validation_step(self, data, target_data):
        batch_size = list(data.values())[0].size(dim=0)

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, batch_size)
        
        return loss, Counter(tqdm_dict)




# Affect
class AffectGMC(SuperGMC):
    def __init__(self, name, exclude_modality, common_dim, latent_dim, infonce_temperature, loss_type="infonce", scenario='mosei'):
        super(AffectGMC, self).__init__(name, common_dim, exclude_modality, latent_dim, infonce_temperature, loss_type)

        if scenario == 'mosei':
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=74, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=35, hidden_dim=30, latent_dim=latent_dim, timestep=50)
        else:
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=5, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=20, hidden_dim=30, latent_dim=latent_dim, timestep=50)

        self.joint_processor = AffectJointProcessor(latent_dim, scenario)

        if exclude_modality == 'vision':
            self.processors = {'text': self.language_processor}
        elif exclude_modality == 'text':
            self.processors = {'vision': self.vision_processor}
        else:
            self.processors = {
                'vision': self.vision_processor,
                'text': self.language_processor,
                'joint': self.joint_processor
            }

        self.loss_type = loss_type
        self.encoder = AffectEncoder(common_dim=common_dim, latent_dim=latent_dim)

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        for key, proc in self.processors.items():
            if key != 'joint':
                proc.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def set_common_dim(self, common_dim):
        self.encoder.set_common_dim(common_dim)
        if "joint" in self.processors.keys():
            self.processors['joint'].set_common_dim(common_dim)
        self.common_dim = common_dim

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality