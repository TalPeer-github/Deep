import math
import numpy as np
import torch
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2
from loss import Attack_Criterion, VOCriterion, test_model
from torch.nn import functional as F
import kornia.geometry as kgm
import kornia.filters as kf
from Datasets.tartanTrajFlowDataset import extract_traj_data
import os
from torchvision.utils import save_image


class Personolize_Attack(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=3,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None):
        super(Personolize_Attack, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                                 sample_window_size, sample_window_stride,
                                                 pert_padding)

        self.last_check_point_loss = None
        self.last_check_point_pert = None
        self.last_check_point_pert_diffs = 0.
        self.prev_train_loss_tot = 0.
        self.lr_list = torch.linspace(0, 1, 30).tolist()
        self.alpha = alpha
        self.n_iter =  n_iter
        self.n_restarts = 1
        self.step_size_just_changed = False
        self.early_stopping = 20
        self.decay_factor = 1.0
        self.eps = 1
        self.rand_init = rand_init
        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                                scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2,
                                device=None):

        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                                   intrinsic_I0,
                                                                   img1_delta, img2_delta,
                                                                   scale,
                                                                   mask1, mask2,
                                                                   perspective1,
                                                                   perspective2,
                                                                   device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        # del loss_sum
        torch.cuda.empty_cache()

        return grad, loss_sum

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                               scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2,
                               device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]
        window_acc_loss = None

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window, window_acc_loss = self.calc_sample_grad_single(pert_window,
                                                                        img1_I0_window,
                                                                        img2_I0_window,
                                                                        intrinsic_I0_window,
                                                                        img1_delta_window,
                                                                        img2_delta_window,
                                                                        scale_window,
                                                                        y_window,
                                                                        clean_flow_window,
                                                                        target_pose_window,
                                                                        perspective1_window,
                                                                        perspective2_window,
                                                                        mask1_window,
                                                                        mask2_window,
                                                                        device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()

        return grad.to(device), window_acc_loss

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):

        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad, _ = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad

            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()

        with torch.no_grad():
            pert += (a_abs * multiplier) * grad_tot
            pert = self.project(pert, eps)

        return pert

    def new_gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, step_size, eps, prev_pert, device=None):
        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)
        loss_tot = 0
        alpha = 0.75
        prev_pert = prev_pert
        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad, loss = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                               img1_delta, img2_delta,
                                               scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                               perspective1, perspective2,
                                               mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad
                loss_tot += loss

            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()

        with torch.no_grad():
            grad_tot = self.normalize_grad(grad_tot)
            # next_pert = pert + grad_tot * self.alpha * multiplier
            # next_pert = self.project(next_pert,eps)
            # self.alpha = self.alpha * multiplier
            z = pert + (step_size * multiplier) * grad_tot
            z = self.project(z, eps)
            x = pert + (alpha * (z - pert)) + ((1 - alpha) * (pert - prev_pert))
            pert = self.project(x, eps)

        return pert, loss_tot

    def compute_checkpoints(self):
        prev_p = 0
        p = 0.12
        checkpoints = [math.ceil(p * self.n_iter)]
        for k in range(1, self.n_iter):
            new_p = p + max(p - prev_p - 0.03, 0.06)
            new_w = math.ceil(new_p * self.n_iter)
            if new_w > self.n_iter:
                break
            checkpoints.append(new_w)
            prev_p = p
            p = new_p
        print(checkpoints)
        return checkpoints

    def update_step_size(self, k_iteration, train_loss_tot, improvements, check_point_prev_loss, step_size,checkpoints,check_point_index):
        fraction = 0.75
        step_size_changed = self.step_size_just_changed
        checkpoint_last_changed = self.checkpoint_last_changed
        first_condition = improvements < fraction * (k_iteration - checkpoints[check_point_index-1])
        second_condition = (not step_size_changed) and (check_point_prev_loss == train_loss_tot)

        step_size_changing = first_condition or second_condition
        self.step_size_just_changed = step_size_changing
        step_size = step_size / 2 if step_size_changing else step_size
        self.checkpoint_last_changed = k_iteration if step_size_changing else checkpoint_last_changed
        return step_size, step_size_changing

    def perturb(self, data_loader, y_list, eps,
                targeted=False, device=None, eval_data_loader=None, eval_y_list=None):  # pgd_linf_rand
        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing APGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)
        pert = torch.zeros_like(best_pert)
        self.alpha = a_abs
        check_point_prev_loss = float('-inf')
        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()
            if self.last_check_point_pert is not None:
                print(" perturbation initialized from checkpoint last pert")
                pert = self.last_check_point_pert
            elif self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")
            pert = self.project(pert, eps)
            train_no_improvements = 0
            eval_no_improvements = 0
            train_improvements = 0
            eval_improvements = 0
            train_best_pert = pert
            train_best_loss = 0
            train_last_improvement = 0
            eval_last_improvement = 0
            step_size_records = {}
            last_eval_loss = 0
            # initial_pert = self.gradient_ascent_step(pert=pert, data_shape=data_shape, data_loader=data_loader, y_list=y_list,
            #                                            clean_flow_list=clean_flow_list,
            #                                            multiplier=multiplier, a_abs=self.alpha, eps=eps, device=device)
            # first_pert, first_loss = self.new_gradient_ascent_step(initial_pert, data_shape, data_loader, y_list,
            #                                                    clean_flow_list,
            #                                                    multiplier, step_size=self.alpha, eps=eps, device=device,
            #                                                    prev_pert=pert)
            # second_pert, second_loss = self.new_gradient_ascent_step(first_pert, data_shape, data_loader, y_list,
            #                                                      clean_flow_list,
            #                                                      multiplier, step_size=self.alpha, eps=eps, device=device,
            #                                                      prev_pert=initial_pert)
            # if second_loss > first_loss:
            #     prev_pert = first_pert
            #     pert = second_pert
            #     train_best_loss = second_loss
            # else:
            #     prev_pert = initial_pert
            #     pert = first_pert
            #     train_best_loss = first_loss
            step_size = self.alpha
            step_size_records["start"] = step_size
            check_point_index = 0
            didnt_visit_no_improvement_flag = True
            checkpoints = self.compute_checkpoints()
            self.checkpoint_last_changed = 0
            self.step_size_just_changed = False
            self.last_check_point_loss = 0
            prev_pert = torch.zeros_like(best_pert)
            for k in tqdm(range(self.n_iter)):
                print("attack optimization epoch: " + str(k))
                pert = pert.clone().detach()
                iter_start_time = time.time()
                new_pert, train_loss_tot = self.new_gradient_ascent_step(pert, data_shape, data_loader, y_list,
                                                                     clean_flow_list,
                                                                     multiplier, step_size, eps, device=device,
                                                                     prev_pert=prev_pert)
                prev_pert = pert
                pert = new_pert
                if train_loss_tot >= train_best_loss:
                    train_best_pert = new_pert
                    train_best_loss = train_loss_tot
                    train_improvements += 1
                    train_no_improvements = 0
                    train_last_improvement = k
                else:
                    train_no_improvements += 1

                # if k in checkpoints:
                #     check_point_index += 1
                #     print(f"Checking checkpoint -- > {k}")
                #     step_size, self.step_size_just_changed = self.update_step_size(k, train_loss_tot,
                #                                                                    train_improvements,
                #                                                                    self.last_check_point_loss,
                #                                                                    step_size,checkpoints=checkpoints,check_point_index=check_point_index)
                #     if self.step_size_just_changed:
                #         step_size_records[f"step_size_{check_point_index}_change"] = step_size
                #         print(f"step_size changes at checkpoint {k} to -- > {step_size} ")
                #         self.last_check_point_loss = train_best_loss
                #         self.last_check_point_pert = train_best_pert.clone().detach()
                #         pert = self.last_check_point_pert
                #         train_improvements = 0

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))
                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)

                    if eval_loss_tot > best_loss_sum:
                        best_pert = pert.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                        eval_improvements += 1
                        eval_no_improvements = 0
                        eval_last_improvement = k
                    else:
                        eval_no_improvements += 1
                        if didnt_visit_no_improvement_flag:
                            self.alpha = step_size
                            didnt_visit_no_improvement_flag = False

                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)


                    if k in checkpoints:
                        check_point_index += 1
                        print(f"Checking checkpoint -- > {k}")
                        step_size, self.step_size_just_changed = self.update_step_size(k, best_loss_sum,
                                                                                       eval_improvements,
                                                                                       self.last_check_point_loss,
                                                                                       step_size,
                                                                                       checkpoints=checkpoints,
                                                                                       check_point_index=check_point_index)
                        self.last_check_point_loss = best_loss_sum
                        if self.step_size_just_changed:
                            step_size_records[f"step_size_{check_point_index}_change"] = step_size
                            print(f"step_size changes at checkpoint {k} to -- > {step_size} ")
                            self.last_check_point_pert = best_pert.clone().detach()
                            pert = self.last_check_point_pert
                            eval_improvements = 0

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))

                    last_eval_loss = eval_loss_tot
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()



                    if train_last_improvement - eval_last_improvement >= 10 or eval_no_improvements >= self.early_stopping:
                        break

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
            pert_dir = f"Best_Pert_Eval_Loss{last_eval_loss}_(alpha:{self.alpha})"
            print(f' saving perturbation to {pert_dir}')
            if not os.path.exists(pert_dir):
                os.makedirs(pert_dir)
            save_image(tensor=best_pert, fp=pert_dir + '/pert_' + str(last_eval_loss) + '_' + '.png')
            del last_eval_loss
            self.init_pert_path = pert_dir
            self.init_pert = best_pert
            print("----- Step Sizes changes are:\n", step_size_records)
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss, traj_clean_loss_mean_list, traj_loss_mean_list, traj_best_loss_mean_list
