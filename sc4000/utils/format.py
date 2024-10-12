def format_args(args):
    if args is None or args == "{}" or args == "":
        return {}
    args_dict = {}
    for arg in args.split(","):
        key, value = arg.split("=")
        args_dict[key] = value
    return args_dict
