import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from torch.optim import Adam
from calvin_base_model import CalvinBaseModel
from datasets.natsgd_data_module import NatSgdDataModule

from decoders.action_decoder import ActionDecoder
from concat_encoders import ConcatEncoders
from plan_proposal_net import PlanProposalNetwork
from plan_recognition_net import PlanRecognitionNetwork
from logistic_policy_network import LogisticPolicyNetwork



from goal_encoders import LanguageGoalEncoder, VisualGoalEncoder
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D
import hydra
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
#logger = logging.getLogger(__name__)


class MCIL(pl.LightningModule, CalvinBaseModel):
    """
    The lightning module used for training.

    Args:
        perceptual_encoder: DictConfig for perceptual_encoder.
        plan_proposal: DictConfig for plan_proposal network.
        plan_recognition: DictConfig for plan_recognition network.
        language_goal: DictConfig for language_goal encoder.
        visual_goal: DictConfig for visual_goal encoder.
        action_decoder: DictConfig for action_decoder.
        kl_beta: Weight for KL loss term.
        optimizer: DictConfig for optimizer.
    """
    #@hydra.main(config_path="../../conf", config_name="config")
    def __init__(
        self,
        #cfg: DictConfig,
        # perceptual_encoder: {'vision_static':"default", 'vision_gripper':None, ""},
        # plan_proposal: DictConfig,
        # plan_recognition: DictConfig,
        # visual_goal: DictConfig,
        # language_goal: DictConfig,
        # action_decoder: DictConfig,
        kl_beta= 0.001,
        optimizer={},
        replan_freq: int = 1,
    ):

        super(MCIL, self).__init__()
        #TODO
        self.perceptual_encoder = ConcatEncoders()#hydra.utils.instantiate(cfg.model.perceptual_encoder)
        
        # plan networks
        #TODO
        self.plan_proposal = PlanProposalNetwork()#hydra.utils.instantiate(plan_proposal)
        self.plan_recognition = PlanRecognitionNetwork()#hydra.utils.instantiate(plan_recognition)

        #goal encoders
        
        #self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = LanguageGoalEncoder()#hydra.utils.instantiate(language_goal) if language_goal else None
        self.gesture_goal = VisualGoalEncoder()
        #policy network
        self.action_decoder: ActionDecoder =  LogisticPolicyNetwork()#hydra.utils.instantiate(action_decoder)

        self.kl_beta = kl_beta
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        # workaround to resolve hydra config file before calling save_hyperparams  until they fix this issue upstream
        # without this, there is conflict between lightning and hydra
        #action_decoder.out_features = 16#action_decoder.out_features

        self.optimizer_config["lr"] = 0.0001
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        self.lang_embeddings = None

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
        """
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters())#hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        self.trainer.reset_train_dataloader(self)
        return optimizer

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution
    ]:
        """
        Main forward pass for training step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            train_acts: Ground truth actions.

        Returns:
            kl_loss: KL loss
            action_loss: Behavior cloning action loss.
            total_loss: Sum of kl_loss and action_loss.
            pp_dist: Plan proposal distribution.
            pr_dist: Plan recognition distribution
        """
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        action_loss = self.action_decoder.loss(sampled_plan, perceptual_emb, latent_goal, train_acts)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        total_loss = action_loss + kl_loss

        return kl_loss, action_loss, total_loss, pp_dist, pr_dist

    def lmp_val(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Main forward pass for validation step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            actions: Groundtruth actions.

        Returns:
            sampled_plan_pp: Plan sampled from plan proposal network.
            action_loss_pp: Behavior cloning action loss computed with plan proposal network.
            sampled_plan_pr: Plan sampled from plan recognition network.
            action_loss_pr: Behavior cloning action loss computed with plan recognition network.
            kl_loss: KL loss
            mae_pp: Mean absolute error (L1) of action sampled with input from plan proposal network w.r.t ground truth.
            mae_pr: Mean absolute error of action sampled with input from plan recognition network w.r.t ground truth.
            gripper_sr_pp: Success rate of binary gripper action sampled with input from plan proposal network.
            gripper_sr_pr: Success rate of binary gripper action sampled with input from plan recognition network.
        """
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------ Policy network ------------ #
        sampled_plan_pp = pp_dist.sample()  # sample from proposal net
        action_loss_pp, sample_act_pp = self.action_decoder.loss_and_act(
            sampled_plan_pp, perceptual_emb, latent_goal, actions
        )

        mae_pp = torch.nn.functional.l1_loss(
            sample_act_pp[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pp = torch.mean(mae_pp, 1)  # (batch, 6)
        # gripper action
        # gripper_discrete_pp = sample_act_pp[..., -1]
        # gt_gripper_act = actions[..., -1]
        # m = gripper_discrete_pp > 0
        # gripper_discrete_pp[m] = 1
        # gripper_discrete_pp[~m] = -1
        # gripper_sr_pp = torch.mean((gt_gripper_act == gripper_discrete_pp).float())

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan_pr = pr_dist.sample()  # sample from recognition net
        action_loss_pr, sample_act_pr = self.action_decoder.loss_and_act(
            sampled_plan_pr, perceptual_emb, latent_goal, actions
        )
        mae_pr = torch.nn.functional.l1_loss(
            sample_act_pr[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pr = torch.mean(mae_pr, 1)  # (batch, 6)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        # gripper action
        # gripper_discrete_pr = sample_act_pr[..., -1]
        # m = gripper_discrete_pr > 0
        # gripper_discrete_pr[m] = 1
        # gripper_discrete_pr[~m] = -1
        # gripper_sr_pr = torch.mean((gt_gripper_act == gripper_discrete_pr).float())

        return (
            sampled_plan_pp,
            action_loss_pp,
            sampled_plan_pr,
            action_loss_pr,
            kl_loss,
            mae_pp,
            mae_pr
        )

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        print("Training Step Start")
        kl_loss, action_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        for tester, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["start"], dataset_batch["end"], dataset_batch["bbox"]
            )
            if torch.sum(dataset_batch['lang']) == 0: #if "lang" in self.modality_scope:
                latent_goal = self.gesture_goal(dataset_batch["end"])
            else:
                latent_goal = self.language_goal(dataset_batch["lang"])#self.perceptual_encoder(dataset_batch["gesture"])
                #latent_goal = self.gesture_goal(gest_emb)
            kl, act_loss, mod_loss, pp_dist, pr_dist = self.lmp_train(
                perceptual_emb, latent_goal, dataset_batch["actions"])
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            # self.log(f"train/kl_loss_scaled_{self.modality_scope}", kl, on_step=False, on_epoch=True)
            # self.log(f"train/action_loss_{self.modality_scope}", act_loss, on_step=False, on_epoch=True)
            # self.log(f"train/total_loss_{self.modality_scope}", mod_loss, on_step=False, on_epoch=True)
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        # self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True)
        # self.log("train/action_loss", action_loss, on_step=False, on_epoch=True)
        # self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        print("Training Step End -- Loss: ", total_loss)
        return total_loss

    def compute_kl_loss(
        self, pr_dist: torch.distributions.Distribution, pp_dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition and plan proposal network.

        Args:
            pr_dist: Distribution produced by plan recognition network.
            pp_dist: Distribution produced by plan proposal network.

        Returns:
            Scaled KL loss.
        """
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int):  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing losses and the sampled plans of plan recognition and plan proposal networks.
        """
        output = {}
        for dataset_batch in batch:
            outputPlan = self.step(dataset_batch)
            similarity = torch.cosine_similarity(outputPlan[0][0], dataset_batch['actions'][0], dim=0)
            print(similarity.item())
            similarity = 0

            # rawDiff = torch.sub(outputPlan, dataset_batch['actions'])
            # avgNonZeroMotors = torch.count_nonzero(outputPlan).item()# + torch.count_nonzero(outputPlan)
            # absTotalDiff = torch.sum(torch.abs(rawDiff)).item()
            # print(f"Average diff: {absTotalDiff/avgNonZeroMotors}")
            # absTotalDiff = 0
            # avgNonZeroMotors = 0
            # rawDiff = 0
            # outputPlan = 0


            # perceptual_emb = self.perceptual_encoder(
            #     dataset_batch["start"], dataset_batch["end"], dataset_batch["bbox"]
            # )

            # if torch.sum(dataset_batch['gesture']) == 0: #if "lang" in self.modality_scope:
            #     latent_goal = self.language_goal(dataset_batch["lang"])
            # else:
            #     latent_goal = self.gesture_goal(dataset_batch["gesture"])
                
            # (
            #     sampled_plan_pp,
            #     action_loss_pp,
            #     sampled_plan_pr,
            #     action_loss_pr,
            #     kl_loss,
            #     mae_pp,
            #     mae_pr
            # ) = self.lmp_val(perceptual_emb, latent_goal, dataset_batch["actions"])
            # # output[f"val_action_loss_pp_"] = action_loss_pp
            # output[f"sampled_plan_pp_"] = sampled_plan_pp
            # print(sampled_plan_pp.shape)
            # # output[f"val_action_loss_pr_"] = action_loss_pr
            # output[f"sampled_plan_pr_"] = sampled_plan_pr
            # print(sampled_plan_pr.shape)
            # # output[f"kl_loss_"] = kl_loss
            # # output[f"mae_pp_"] = mae_pp
            # # output[f"mae_pr_"] = mae_pr

        return output

    def validation_epoch_end(self, validation_step_outputs):
        print("")

        # val_total_act_loss_pr = torch.tensor(0.0).to(self.device)
        # val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        # val_kl_loss = torch.tensor(0.0).to(self.device)
        # val_total_mae_pr = torch.tensor(0.0).to(self.device)
        # val_total_mae_pp = torch.tensor(0.0).to(self.device)
        # val_pos_mae_pp = torch.tensor(0.0).to(self.device)
        # val_pos_mae_pr = torch.tensor(0.0).to(self.device)
        # val_orn_mae_pp = torch.tensor(0.0).to(self.device)
        # val_orn_mae_pr = torch.tensor(0.0).to(self.device)
        # for mod in self.trainer.datamodule.modalities:
        #     act_loss_pp = torch.stack([x[f"val_action_loss_pp_{mod}"] for x in validation_step_outputs]).mean()
        #     act_loss_pr = torch.stack([x[f"val_action_loss_pr_{mod}"] for x in validation_step_outputs]).mean()
        #     kl_loss = torch.stack([x[f"kl_loss_{mod}"] for x in validation_step_outputs]).mean()
        #     mae_pp = torch.cat([x[f"mae_pp_{mod}"] for x in validation_step_outputs])
        #     mae_pr = torch.cat([x[f"mae_pr_{mod}"] for x in validation_step_outputs])
        #     pr_mae_mean = mae_pr.mean()
        #     pp_mae_mean = mae_pp.mean()
        #     pos_mae_pp = mae_pp[..., :3].mean()
        #     pos_mae_pr = mae_pr[..., :3].mean()
        #     orn_mae_pp = mae_pp[..., 3:6].mean()
        #     orn_mae_pr = mae_pr[..., 3:6].mean()
        #     val_total_mae_pr += pr_mae_mean
        #     val_total_mae_pp += pp_mae_mean
        #     val_pos_mae_pp += pos_mae_pp
        #     val_pos_mae_pr += pos_mae_pr
        #     val_orn_mae_pp += orn_mae_pp
        #     val_orn_mae_pr += orn_mae_pr
        #     val_total_act_loss_pp += act_loss_pp
        #     val_total_act_loss_pr += act_loss_pr
        #     val_kl_loss += kl_loss

        #     self.log(f"val_act/{mod}_act_loss_pp", act_loss_pp, sync_dist=True)
        #     self.log(f"val_act/{mod}_act_loss_pr", act_loss_pr, sync_dist=True)
        #     self.log(f"val_total_mae/{mod}_total_mae_pr", pr_mae_mean, sync_dist=True)
        #     self.log(f"val_total_mae/{mod}_total_mae_pp", pp_mae_mean, sync_dist=True)
        #     self.log(f"val_pos_mae/{mod}_pos_mae_pr", pos_mae_pr, sync_dist=True)
        #     self.log(f"val_pos_mae/{mod}_pos_mae_pp", pos_mae_pp, sync_dist=True)
        #     self.log(f"val_orn_mae/{mod}_orn_mae_pr", orn_mae_pr, sync_dist=True)
        #     self.log(f"val_orn_mae/{mod}_orn_mae_pp", orn_mae_pp, sync_dist=True)
        #     self.log(f"val_kl/{mod}_kl_loss", kl_loss, sync_dist=True)
        # self.log(
        #     "val_act/action_loss_pp", val_total_act_loss_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        # )
        # self.log(
        #     "val_act/action_loss_pr", val_total_act_loss_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        # )
        # self.log("val_kl/kl_loss", val_kl_loss / len(self.trainer.datamodule.modalities), sync_dist=True)
        # self.log(
        #     "val_total_mae/total_mae_pr", val_total_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        # )
        # self.log(
        #     "val_total_mae/total_mae_pp", val_total_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        # )
        # self.log("val_pos_mae/pos_mae_pr", val_pos_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        # self.log("val_pos_mae/pos_mae_pp", val_pos_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        # self.log("val_orn_mae/orn_mae_pr", val_orn_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        # self.log("val_orn_mae/orn_mae_pp", val_orn_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        # replan every replan_freq steps (default 30 i.e every second)

        strOrGest = True
        if torch.sum(obs['gesture']) == 0: #if "lang" in self.modality_scope:
            goal = obs["lang"]
        else:
            goal = obs["gesture"]
            strOrGest = False

        if strOrGest:
            self.plan, self.latent_goal = self.get_pp_plan_lang(obs, goal)
        else:
            self.plan, self.latent_goal = self.get_pp_plan_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict_with_plan(obs, self.latent_goal, self.plan)
        return action

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}

    def predict_with_plan(
        self,
        obs: Dict[str, Any],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass observation, goal and plan through decoder to get predicted action.

        Args:
            obs: Observation from environment.
            latent_goal: Encoded goal.
            sampled_plan: Sampled plan proposal plan.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["start"], obs["end"], obs["bbox"])
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_pp_plan_vision(self, obs: dict, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Goal observation (vision & proprioception).

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded visual goal.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["start"], obs["end"], obs["bbox"])
            latent_goal = self.gesture_goal(goal)
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            sampled_plan = pp_dist.sample()  # sample from proposal net
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(self, obs: dict, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Embedded language instruction.

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded language goal.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["start"], obs["end"], obs["bbox"])
            latent_goal = self.language_goal(goal)
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            sampled_plan = pp_dist.sample()  # sample from proposal net
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        print(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        print(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        print(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        print(f"Finished validation epoch {self.current_epoch}")

def train_logger():
    print("Epoch Done")

#@hydra.main(config_path="../../conf", config_name="config")
#def build_model(cfg: DictConfig):
 #   model = hydra.utils.instantiate(cfg.model)

if __name__ == "__main__":
    print("Here1")
    datasets = np.load('NatSGD_Dataset_v0.9.0e.npz', allow_pickle=True)['NatSGD_Dataset']
    filtered_dataset = np.array([x for x in datasets if np.array(x[12]).shape[0] <= 175])
    indexes = [i for i in range(54)]
    print("Initial Shape: ", filtered_dataset.shape)
    val = [31, 32, 33, 34, 35]
    validation = []
    training = []
    for v in val:
        print("Index: ", v)
        validation.append(datasets[v])
    training = filtered_dataset
    validation = np.array(validation)
    print("VAL: ", validation.shape, "train: ", training.shape)
    data = NatSgdDataModule(datasets = {'training':training, 'validation':validation})
    data.setup(unsplit_dataset=filtered_dataset)
    train_data_loader = data.train_dataloader()
    # for batch in train_data_loader:
    #     data = batch['data']
    #     target = batch['target']
    #     print(data.shape)

    val_data_loader = data.val_dataloader()
    trainer_args = {
    "devices":1,
    "accelerator":"auto",
    "precision":16,
    "val_check_interval":1.0,
    "max_epochs":50,
    "benchmark":False,
    "log_every_n_steps":200

    #"train_logger":train_logger()
    }
    trainer = Trainer(**trainer_args)
    p_e = {}
    #model = hydra.utils.instantiate()
    model = MCIL()
    trainer.fit(model, train_data_loader, val_data_loader)
    trainer.save_checkpoint('./trained_models/firstmodel.ckpt')
    print(data)