from cs336_basics.lm_trainer import (
    planned_checkpoint_steps,
    resolve_checkpointing_every,
    should_save_checkpoint,
)


def test_checkpointing_every_overrides_save_interval():
    steps = planned_checkpoint_steps(
        total_iterations=10,
        save_interval=3,
        checkpointing_every=4,
    )
    assert steps == [4, 8, 9]


def test_save_interval_backward_compatibility_when_override_missing():
    steps = planned_checkpoint_steps(
        total_iterations=10,
        save_interval=3,
        checkpointing_every=None,
    )
    assert steps == [3, 6, 9]


def test_checkpointing_every_non_positive_disables_periodic_but_keeps_final():
    steps = planned_checkpoint_steps(
        total_iterations=10,
        save_interval=3,
        checkpointing_every=0,
    )
    assert steps == [9]


def test_final_checkpoint_always_saved():
    periodic = resolve_checkpointing_every(save_interval=10, checkpointing_every=0)
    assert should_save_checkpoint(4, total_iterations=5, checkpointing_every=periodic)
