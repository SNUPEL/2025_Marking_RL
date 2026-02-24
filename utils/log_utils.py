def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, loss, bl_loss, avg_cost_nn, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # # Log values to screen
    # print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))
    #
    # print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        tb_logger.log_value('avg_cost_nn', avg_cost_nn, step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

    if not opts.no_vessl:
        import vessl

        vessl.log(payload={"avg_cost": avg_cost}, step=step)

        vessl.log(payload={"actor_loss": loss.item()}, step=step)
        vessl.log(payload={"nll": -log_likelihood.mean().item()}, step=step)

        vessl.log(payload={"grad_norm": grad_norms[0]}, step=step)
        vessl.log(payload={"grad_norm_clipped": grad_norms_clipped[0]}, step=step)

        vessl.log(payload={'avg_cost_nn': avg_cost_nn}, step=step)

        if opts.baseline == 'critic':
            vessl.log(payload={"critic_loss": bl_loss.item()}, step=step)
            vessl.log(payload={"critic_grad_norm": grad_norms[1]}, step=step)
            vessl.log(payload={"critic_grad_norm_clipped": grad_norms_clipped[1]}, step=step)