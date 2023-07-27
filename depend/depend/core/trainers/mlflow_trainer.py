
        run_name = '/'.join([self.project_name, model_name])
        if self.config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": self.config.train.tags + ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
                elif config.train.tracker == "tensorboard":
                    # flatten config for tensorboard, split list in hparams into flatten config
                    if config_dict["model"].get("peft_config", None):  # tensorboard does not support peft config type
                        config_dict["model"]["peft_config"] = str(config_dict["model"]["peft_config"])
                    config_dict_flat = flatten_dict(config_dict)
                    config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat["optimizer/kwargs/betas"][0]
                    config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat["optimizer/kwargs/betas"][1]
                    config_dict_flat.pop("optimizer/kwargs/betas", None)
                    for ix, tag in enumerate(config_dict_flat.pop("train/tags")):
                        config_dict_flat[f"train/tag_{ix}"] = tag

                    self.accelerator.init_trackers(
                        project_name=self.config.train.project_name,
                        config=config_dict_flat,
                    )
                elif config.train.tracker is None:
                    self.accelerator.init_trackers(project_name=self.config.train.project_name)
                else:
                    raise ValueError(
                        f"Only supported trackers are `wandb` and `tensorboard`. Got: `{config.train.tracker}`. "
                        "Set `tracker` to `None` to disable tracking."
                    )

            self.nth_evaluation = 0
            self.generate_sweep_kwarg = None
            for k, v in self.config.method.gen_kwargs.items():
                if isinstance(v, list):
                    if self.generate_sweep_kwarg is not None:
                        logger.info("Only a single sweep is allowed, {k} is going to be set to {v[0]}")
                        self.generate_kwargs[k] = v[0]
                    else:
                        self.generate_sweep_kwarg = (k, v)


    def run(self):
        with mlflow.start_run as run:
            for k, v in self.train_kwargs.items():
                mlflow.log_param(k, v)
            for epoch in range(self.epochs):
                with catchtime() as t:
                    # training
                    train_info = self.train_one_epoch()
                    
                    self.logger.epoch_info(Epoch = epoch + 1, Time = '%fs' % float(t_))
                    train_info = self.train()
                    self.weak_line('train info')
                    self.logger.info(**train_info)
                    
                    test_info = self.test()
                    self.weak_line('test info')
                    self.logger.info(**test_info)

                    self.strong_line()
                    
                    # evaluate with deterministic policy

                    
                    tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
                    tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
                    tb_writer.add_scalar("Eval_L-deterministic", L, epoch)
                    agent.policy.use_mean = False
                    mlflow.log_metric(k, v, epoch)

                    # evaluate with stochastic policy
                    dataset = core.evaluate(n_episodes=n_eval_episodes)
                    R_mean_stoch = np.mean(compute_J(dataset))
                    J_mean_stoch = np.mean(compute_J(dataset, gamma=gamma))
                    L = np.mean(compute_episodes_length(dataset))
                    self.logger.log_numpy(Epoch=epoch, R_mean=R_mean_stoch, J_mean=J_mean_stoch, L=L)
                    tb_writer.add_scalar("Eval_R-stochastic", R_mean_stoch, epoch)
                    tb_writer.add_scalar("Eval_J-stochastic", J_mean_stoch, epoch)
                    tb_writer.add_scalar("Eval_L-stochastic", L, epoch)
                    mlflow.log_metric(k, v, epoch)
                    print("R_mean (deter): %f | R_mean (stoch): %f" % (R_mean, R_mean_stoch))

                    # save agent if needed
                    agent_saver.save(core.agent, J_mean)

            agent_saver.save_curr_best_agent()
            print("Finished.")

            
                # print out active_run
                logger.info("Active Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))

                
            logger.info("Uploading TensorFlow events as a run artifact.")
            mlflow.log_artifacts(log_dir, artifact_path="events")
    
    

