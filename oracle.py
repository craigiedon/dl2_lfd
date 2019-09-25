import numpy as np
import torch
from constraints import constraint_loss

def evaluate_constraint(input_batch, target_batch, constraint, args):
    learning_rate = 0.01
    if constraint.n_gvars > 0:
        domains = constraint.get_domains(input_batch, target_batch)
        z_batches = general_attack(input_batch, target_batch, constraint, domains,
                                   num_restarts=1, num_iters=args["num_iters"], learning_rate=learning_rate, args=args)
    else:
        z_batches = None

    neg_losses, pos_losses , is_satisfied, _ = constraint_loss(constraint, input_batch, target_batch, z_batches)

    if not isinstance(is_satisfied, np.ndarray):
        is_satisfied = is_satisfied.cpu().numpy()

    constraints_acc = np.mean(is_satisfied)
    return torch.mean(pos_losses), constraints_acc


def general_attack(input_batch, target_batch, constraint, domains, num_restarts, num_iters, learning_rate, args):
    batch_size = target_batch.size()[0]
    num_global_vars = len(domains)

    for _ in range(num_restarts):
        z_batches = [dom.sample() for dom in domains]
        z_ins = [z_batch.clone() for z_batch in z_batches]
        for z_in in z_ins:
            z_in.requires_grad = True

        # This has shape: inp_batch_size x num_global_variables
        # Should have shape num_global_variables x inp_batch_size

        assert len(z_batches) == num_global_vars
        assert z_batches[0].shape[0] == batch_size
        assert z_batches[0].ndim == 2

        for itr in range(num_iters):
            # print(itr)
            neg_losses, _, _, z_inputs = constraint_loss(constraint, input_batch, target_batch, z_ins)

            avg_neg_loss = torch.mean(neg_losses)
            avg_neg_loss.backward()
            # This part is Projected Gradient Descent:
            # Do normal SGD, then project result within bounds
            for g_var_ind in range(num_global_vars):
                # print(z_ins[g_var_ind].grad)
                # print("Gradz: ", z_ins[g_var_ind].grad)
                z_batches[g_var_ind] -= learning_rate * z_ins[g_var_ind].grad

            # Project all z values according to domain constraints
            for g_var_ind in range(num_global_vars):
                # for batch_num in range(batch_size):
                z_batches[g_var_ind] = domains[g_var_ind].project(z_batches[g_var_ind])

        return z_batches # TODO: support multiple retries
