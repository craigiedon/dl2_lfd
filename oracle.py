import numpy as np
import torch

def evaluate_constraint(input_batch, target_batch, constraint, args):
    learning_rate = 0.01
    if constraint.n_gvars > 0:
        domains = constraint.get_domains(input_batch, target_batch)
        z_batches = general_attack(input_batch, target_batch, constraint, domains,
                                   num_restarts=1, num_iters=args.num_iters, learning_rate=learning_rate, args=args)
    else:
        z_batches = None

    neg_losses, _ , is_satisfied, _ = constraint.loss(input_batch, target_batch, z_batches, args)

    if not isinstance(is_satisfied, np.ndarray):
        is_satisfied = is_satisfied.cpu().numpy()

    constraints_acc = np.mean(is_satisfied)
    return torch.mean(neg_losses), constraints_acc


def general_attack(input_batch, target_batch, constraint, domains, num_restarts, num_iters, learning_rate, args):
    inp_batch_size = input_batch.size()[0]
    num_global_vars = len(domains)

    # TODO: Is this the desired shape?
    for _ in range(num_restarts):
        z_batches = np.array([[domains[j][i].sample() for i in range(inp_batch_size)] 
                     for j in range(num_global_vars)])

        # This has shape: inp_batch_size x num_global_variables
        # Should have shape num_global_variables x inp_batch_size

        assert z_batches.shape[0] == num_global_vars
        assert z_batches.shape[1] == inp_batch_size

        for _ in range(num_iters):
            neg_losses, _, _, z_inputs = constraint.loss(input_batch, target_batch, z_batches, args)

            avg_neg_loss = torch.mean(neg_losses)
            avg_neg_loss.backward()
            # This part is Projected Gradient Descent:
            # Do normal SGD, then project result within bounds
            for g_var_ind in range(num_global_vars):
                z_grad = z_inputs[g_var_ind].grad.data
                z_grad = z_grad.cpu().numpy()
                z_batches[g_var_ind] -= learning_rate * z_grad

            # Project all z values according to domain constraints
            for g_var_ind in range(num_global_vars):
                for batch_num in range(inp_batch_size):
                    z_batches[g_var_ind][batch_num] = domains[g_var_ind][batch_num].project(z_batches[g_var_ind][batch_num])

        return z_batches # TODO: support multiple retries
