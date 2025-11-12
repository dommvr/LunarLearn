def apply_bitfit(model):
    for name, p in model.named_parameters():
        pname = name.split(".")[-1]
        p.requires_grad = pname == "b"
        
    return model