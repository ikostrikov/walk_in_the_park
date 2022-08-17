# walk_in_the_park

## Training in simulation
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=A1Run-v0 \
                --utd_ratio=20 \
                --start_training 1000 \
                --max_steps 100000 \
                --config=configs/droq_config.py
```