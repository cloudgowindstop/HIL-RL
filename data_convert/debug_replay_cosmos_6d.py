c factory is not callable: {spec}")
    return factory


def execute_commands(commands: np.ndarray, factory_spec: str, max_steps: int) -> None:
    if not factory_spec:
        raise ValueError("--execute requires --env_factory module:function")
    if max_steps < 1:
        raise ValueError("--execute requires --max_execute_steps >= 1")
    print(f"DANGER: about to execute {min(max_steps, len(commands))} action(s) on real hardware.")
    if input("Type EXECUTE to continue: ").strip() != "EXECUTE":
        raise RuntimeError("robot execution cancelled")
    env = load_factory(factory_spec)()
    try:
        env.reset()
        for command in commands[:max_steps]:
            env.step(command)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--create_debug_parquet", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("pick_spoon_6d_report.csv"))
    parser.add_argument("--translation_scale", type=float, default=0.02)
    parser.add_argument("--rotation_scale", type=float, default=0.06)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--env_factory", default="")
    parser.add_argument("--max_execute_steps", type=int, default=0)
    args = parser.parse_args()

    poses, grippers = load_hdf5_episode(args.hdf5)
    if args.create_debug_parquet:
        actions, clipped = build_actions(poses, grippers, args.translation_scale, args.gripper_scale)
        write_debug_parquet(
            args.parquet, args.hdf5, poses, grippers, actions, clipped,
            args.translation_scale, args.gripper_scale,
        )
        print(f"Wrote 10D action debug Parquet: {args.parquet}")

    actions = parquet_actions(args.parquet)
    rows, commands = validate_round_trip(
        poses, grippers, actions,
        args.translation_scale, args.rotation_scale, args.gripper_scale,
    )
    write_report(args.report, rows, commands)
    summary = {
        "frames": len(rows),
        "max_position_error_m": max(row["position_error_m"] for row in rows),
        "max_rotation_error_deg": max(row["rotation_error_deg"] for row in rows),
        "max_gripper_error": max(row["gripper_error"] for row in rows),
        "translation_clipped_frames": sum(row["translation_clipped"] for row in rows),
        "base_env_rotation_clipped_frames": sum(row["base_env_rotation_clipped"] for row in rows),
        "report": str(args.report),
        "base_env_actions": str(args.report.with_suffix(".base_env_actions.npy")),
    }
    print(json.dumps(summary, indent=2))
    if args.execute:
        execute_commands(commands, args.env_factory, args.max_execute_steps)


if __name__ == "__main__":
    main()
