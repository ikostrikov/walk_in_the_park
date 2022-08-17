from dm_control.rl import control


def _find_non_contacting_height(physics,
                                walker,
                                x_pos=0.0,
                                y_pos=0.0,
                                qpos=None,
                                quat=None,
                                maxiter=1000):
    z_pos = 0.0  # Start embedded in the floor.
    num_contacts = 1
    count = 1
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                if qpos is not None:
                    physics.bind(walker._joints).qpos[:] = qpos
                walker.set_pose(physics, [x_pos, y_pos, z_pos], quat)
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon

        z_pos += 0.01
        count += 1
        if count > maxiter:
            raise ValueError(
                'maxiter reached: possibly contacts in null pose of body.')


def _find_non_contacting_qpos(physics, random_state, walker, maxiter=1000):
    num_contacts = 1
    count = 1
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                walker.set_pose(physics, [0, 0, 100])
                joints_range = physics.bind(walker._joints).range
                qpos = random_state.uniform(joints_range[:, 0],
                                            joints_range[:, 1])
                walker.configure_joints(physics, qpos)
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon

        count += 1

        if count > maxiter:
            raise ValueError(
                'maxiter reached: possibly contacts in null pose of body.')

    walker.set_pose(physics, [0, 0, 0])

    if num_contacts == 0:
        return qpos
    else:
        return None
